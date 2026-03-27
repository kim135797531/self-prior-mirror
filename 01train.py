#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train.py — Training loop, env/model setup, and execute(). Eval-only logic in program.fe_eval."""
import os

# Thread settings for using deterministic numpy algorithms
os.environ["OMP_NUM_THREADS"] = "12"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "12"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "12"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "12"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "12"  # export NUMEXPR_NUM_THREADS=6
import random
import shutil
import warnings


warnings.filterwarnings("ignore", ".*box bound precision lowered.*")

from dataclasses import asdict
import time
import gymnasium
import gymnasium.vector
import torch
import numpy as np
from tensorboardX import SummaryWriter
import argparse
import os
from ruamel.yaml import YAML
import json
from einops import rearrange

from program.envs.env_loader import make_env
from program.episode import EpisodeStore
from program.models.agent import Agent
from program.models.configs.model_config import ModelConfig
from program.fe_trainer import FETrainer
from program.fe_runner import fe_run, run_eval_round
from program.utils import log_video
from script.vacant_gpu import get_vacant_gpu
from script.csv_data import CSVData, ConvertCSVtoHDF5
from program.load_checkpoint import load_config_and_model


def run_training(
    conf,
    np_rng,
    train_env,
    eval_env,
    model,
    writer,
    eval_return_csv,
    eval_free_energy_csv,
    current_step,
    tot_episodes,
    checkpoint,
    is_resume,
):
    """Training setup (trainer, optimizers, episode_store) + training loop. Caller does cleanup."""
    trainer = FETrainer(conf, np_rng, model)
    world_scaler = torch.amp.GradScaler(device=conf.device, enabled=conf.use_amp)
    world_optimizer, world_scheduler = trainer.configure_optimizers(
        optimizer_name="adamw",
        target_modules=[model.obs_provider, model.fe_world],
        learning_rate=conf.world_lr,
        min_learning_rate=conf.min_world_lr,
        use_decay=conf.use_decay_world,
    )
    actor_scaler = torch.amp.GradScaler(device=conf.device, enabled=conf.use_amp)
    actor_optimizer, actor_scheduler = trainer.configure_optimizers(
        optimizer_name="adam",
        target_modules=[model.efe_policy],
        learning_rate=conf.policy_lr,
        min_learning_rate=conf.min_policy_lr,
        use_decay=conf.use_decay_policy,
        eps=conf.policy_adam_eps,
    )
    critic_scaler = torch.amp.GradScaler(device=conf.device, enabled=conf.use_amp)
    critic_optimizer, critic_scheduler = trainer.configure_optimizers(
        optimizer_name="adam",
        target_modules=[model.efe_value],
        learning_rate=conf.policy_lr,
        min_learning_rate=conf.min_policy_lr,
        use_decay=conf.use_decay_policy,
        eps=conf.policy_adam_eps,
    )
    self_prior_scaler = torch.amp.GradScaler(device=conf.device, enabled=conf.use_amp)
    self_prior_optimizer, self_prior_scheduler = trainer.configure_optimizers(
        optimizer_name="adamw",
        target_modules=[model.self_prior],
        learning_rate=conf.self_prior_lr,
        min_learning_rate=conf.min_self_prior_lr,
        use_decay=conf.use_decay_self_prior,
    )

    if is_resume and checkpoint is not None:
        world_scaler.load_state_dict(checkpoint["world_scaler"])
        world_optimizer.load_state_dict(checkpoint["world_optimizer"])
        actor_scaler.load_state_dict(checkpoint["actor_scaler"])
        actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        critic_scaler.load_state_dict(checkpoint["critic_scaler"])
        critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self_prior_scaler.load_state_dict(checkpoint["self_prior_scaler"])
        self_prior_optimizer.load_state_dict(checkpoint["self_prior_optimizer"])
        if "world_scheduler" in checkpoint:
            world_scheduler.load_state_dict(checkpoint["world_scheduler"])
            actor_scheduler.load_state_dict(checkpoint["actor_scheduler"])
            critic_scheduler.load_state_dict(checkpoint["critic_scheduler"])
            self_prior_scheduler.load_state_dict(checkpoint["self_prior_scheduler"])
        if "trainer" in checkpoint:
            trainer.load_checkpoint_state(checkpoint["trainer"])

    episode_store = EpisodeStore(conf, np_rng)
    if is_resume and checkpoint is not None and "episode_store" in checkpoint:
        episode_store.load_state_dict(checkpoint["episode_store"], np_rng)

    if checkpoint is not None:
        del checkpoint

    # Training loop (former mainloop body)
    total_steps = int(conf.total_steps)
    max_episodes = conf.max_episodes
    reconstruction_dict = dict()

    gpu_memory_tested = False

    episode_t0 = time.time()
    done = True
    while current_step < total_steps:
        if tot_episodes >= max_episodes:
            print("Maximum Episodes reached!")
            break

        print("")
        print(f"({'_'.join(conf.config)})")
        print(f"[Seed] {conf.seed}")
        print(f"[Episode] {tot_episodes}")
        print(f"[Train] Start {conf.n_epochs} epochs: ", end="", flush=True)

        ############################################
        # Training world model
        ############################################
        main_loss_dict = dict()
        main_policy_loss_dict = dict()
        reconstruction_dict = dict()

        if current_step == 0 or conf.recon_model_period <= 0:
            log_image_epoch = False
        else:
            end_step = current_step + conf.n_epochs
            last_log_step = end_step - (end_step % conf.recon_model_period)
            if last_log_step > current_step:
                log_image_epoch = conf.n_epochs - (end_step - last_log_step) - 1
            else:
                log_image_epoch = None

        for epoch in range(conf.n_epochs):
            is_log_images = epoch == log_image_epoch

            if episode_store.n_episodes >= conf.train_world_episodes_from:
                # Sample B trajectories of length L from D
                group_keys = [
                    episode_store.get_key(policy_used=False, sticker_used=False),
                    episode_store.get_key(policy_used=False, sticker_used=True),
                    episode_store.get_key(policy_used=True, sticker_used=False),
                    episode_store.get_key(policy_used=True, sticker_used=True),
                ]
                group_probs = [0.25, 0.25, 0.25, 0.25]
                # group_probs = [0.4, 0.4, 0.1, 0.1]

                (path_obs_vision, path_obs_proprio, path_act) = (
                    episode_store.sample_paths(
                        conf.n_paths,
                        conf.n_steps,
                        conf.device,
                        group_keys=group_keys,
                        group_probs=group_probs,
                    )
                )

                # Infer states using the world model
                # Update the world model parameters on the B trajectories, minimizing L_phi
                # (Prior network, posterior network, representation model)
                world_loss_dict, world_sample_dict = trainer.train_world(
                    world_scaler,
                    world_optimizer,
                    world_scheduler,
                    path_obs_vision,
                    path_obs_proprio,
                    path_act,
                    get_reconstruction=is_log_images,
                )
                del (path_obs_vision, path_obs_proprio, path_act)
                reconstruction_dict |= world_sample_dict

                # Add loss to log later
                for k in world_loss_dict:
                    if k in main_loss_dict:
                        main_loss_dict[k] += world_loss_dict[k] / conf.n_epochs
                    else:
                        main_loss_dict[k] = world_loss_dict[k] / conf.n_epochs

            if episode_store.n_episodes >= conf.train_self_prior_episodes_from:
                # Flow is used to estimate the general density with various actions,
                # so it only gets data from random actions. (It will be removed later when homeostasis is added)
                # Also, if there is no caregiver at all, log_prob will be broken, so it simulates putting caregiver experience in rarely
                group_keys = [
                    episode_store.get_key(policy_used=False, sticker_used=False),
                    episode_store.get_key(policy_used=False, sticker_used=True),
                ]
                group_probs = [0.95, 0.05]

                (random_path_obs_vision, random_path_obs_proprio, _) = (
                    episode_store.sample_paths(
                        conf.n_paths,
                        conf.n_steps,
                        conf.device,
                        group_keys=group_keys,
                        group_probs=group_probs,
                    )
                )

                if is_log_images:
                    # Test flow model
                    test_self_prior_n_paths = 4
                    test_self_prior_n_steps = 4
                    obs_no_sticker = episode_store.sample_paths(
                        test_self_prior_n_paths,
                        test_self_prior_n_steps,
                        conf.device,
                        group_keys=[
                            episode_store.get_key(policy_used=False, sticker_used=False)
                        ],
                    )
                    obs_sticker = episode_store.sample_paths(
                        test_self_prior_n_paths,
                        test_self_prior_n_steps,
                        conf.device,
                        group_keys=[
                            episode_store.get_key(policy_used=False, sticker_used=True)
                        ],
                    )
                else:
                    (
                        obs_no_sticker,
                        obs_sticker,
                    ) = (None, None)

                self_prior_loss_dict, self_prior_sample_dict = (
                    trainer.train_self_prior_transformer(
                        self_prior_scaler,
                        self_prior_optimizer,
                        self_prior_scheduler,
                        random_path_obs_vision,
                        random_path_obs_proprio,
                        obs_no_sticker,
                        obs_sticker,
                        get_sample=is_log_images,
                    )
                )
                del (random_path_obs_vision, random_path_obs_proprio)
                if is_log_images:
                    del (obs_no_sticker, obs_sticker)
                reconstruction_dict |= self_prior_sample_dict

                # Add loss to log later
                for k in self_prior_loss_dict:
                    if k in main_loss_dict:
                        main_loss_dict[k] += self_prior_loss_dict[k] / conf.n_epochs
                    else:
                        main_loss_dict[k] = self_prior_loss_dict[k] / conf.n_epochs

            if episode_store.n_episodes >= conf.train_policy_episodes_from:
                # Sample B trajectories of length L from D
                group_keys = [
                    episode_store.get_key(policy_used=False, sticker_used=False),
                    episode_store.get_key(policy_used=False, sticker_used=True),
                    episode_store.get_key(policy_used=True, sticker_used=False),
                    episode_store.get_key(policy_used=True, sticker_used=True),
                ]

                (
                    imagine_path_obs_vision,
                    imagine_path_obs_proprio,
                    imagine_path_act,
                ) = episode_store.sample_paths(
                    conf.n_imagine_paths,
                    conf.n_imagine_context_steps,
                    conf.device,
                    group_keys=group_keys,
                )

                with torch.set_grad_enabled(not conf.use_reinforce):
                    (
                        imagine_policy_feature,
                        imagine_action,
                        imagine_free_energy,
                        imagine_loss_dict,
                        imagine_sample_dict,
                    ) = model.imagine(
                        imagine_path_obs_vision,
                        imagine_path_obs_proprio,
                        imagine_path_act,
                        get_reconstruction=is_log_images,
                    )
                    reconstruction_dict |= imagine_sample_dict

                    del (
                        imagine_path_obs_vision,
                        imagine_path_obs_proprio,
                        imagine_path_act,
                    )

                # Imagine I trajectories of length H from each s
                # Update the action network parameters on the I trajectories, minimizing L_theta
                # Update the expected utility network parameters on the I trajectories, minimizing L_psi
                policy_loss_dict = trainer.train_policy_value(
                    actor_scaler,
                    actor_optimizer,
                    actor_scheduler,
                    critic_scaler,
                    critic_optimizer,
                    critic_scheduler,
                    imagine_policy_feature,
                    imagine_action,
                    imagine_free_energy,
                )

                del (imagine_policy_feature, imagine_action, imagine_free_energy)

                policy_loss_dict |= imagine_loss_dict

                for k in policy_loss_dict:
                    if k in main_policy_loss_dict:
                        main_policy_loss_dict[k] += policy_loss_dict[k] / conf.n_epochs
                    else:
                        main_policy_loss_dict[k] = policy_loss_dict[k] / conf.n_epochs

            if (
                episode_store.n_episodes >= conf.train_world_episodes_from
                or episode_store.n_episodes >= conf.train_self_prior_episodes_from
                or episode_store.n_episodes >= conf.train_policy_episodes_from
            ):
                current_step += 1
                print(f"{epoch}, ", end="", flush=True)
        print("")
        train_t1 = time.time()
        train_time = train_t1 - episode_t0
        print(f"[Train] Trained steps: {current_step}, Time: {train_time:.2f}s")
        print(
            f"[Train] lr: world={world_scheduler.get_last_lr()[0]:.6f} policy={actor_scheduler.get_last_lr()[0]:.6f} self_prior={self_prior_scheduler.get_last_lr()[0]:.6f}"
        )

        # ---------Training epoch end, Logging start---------
        if len(main_loss_dict) > 0:
            for k, v in main_loss_dict.items():
                writer.add_scalar("01_metric_world/" + k, v, global_step=current_step)

        if len(main_policy_loss_dict) > 0:
            for k, v in main_policy_loss_dict.items():
                writer.add_scalar("02_metric_policy/" + k, v, global_step=current_step)

        if len(reconstruction_dict) > 0:
            # Log videos for Likelihood active inference decoder
            # (Not available for contrastive method)
            # print("Logging videos")
            for k, v in reconstruction_dict.items():
                if "self_prior" in k:
                    tag = f"04_render_self_prior/{k}"
                else:
                    tag = f"03_render_world/{k}"
                v = rearrange(v, "t c h w -> 1 t c h w")
                log_video(writer, np_rng, tag, current_step, v)

        # Save model
        if (
            len(main_loss_dict) > 0
            and conf.save_model_period > 0
            and tot_episodes % conf.save_model_period == 0
        ):
            checkpoint = {
                "conf": model.conf,
                "model": model.state_dict(),
                "world_scaler": world_scaler.state_dict(),
                "world_optimizer": world_optimizer.state_dict(),
                "actor_scaler": actor_scaler.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "critic_scaler": critic_scaler.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
                "self_prior_scaler": self_prior_scaler.state_dict(),
                "self_prior_optimizer": self_prior_optimizer.state_dict(),
                "world_scheduler": world_scheduler.state_dict(),
                "actor_scheduler": actor_scheduler.state_dict(),
                "critic_scheduler": critic_scheduler.state_dict(),
                "self_prior_scheduler": self_prior_scheduler.state_dict(),
                "current_step": current_step,
                "tot_episodes": tot_episodes,
                "np_rng_state": np_rng.bit_generator.state,
                "trainer": trainer.get_checkpoint_state(),
            }
            if conf.checkpoint_save_episode_store:
                checkpoint["episode_store"] = episode_store.state_dict()
            if not conf.nolog:
                ckpt_path_num = f"{conf.log_path}/checkpoint_{tot_episodes:05d}.pt"
                ckpt_path_latest = f"{conf.log_path}/checkpoint.pt"
                print(f"Saving model to {ckpt_path_num}")
                torch.save(checkpoint, ckpt_path_num)
                shutil.copy(ckpt_path_num, ckpt_path_latest)

        ########################################################
        # Testing model & Collecting data
        ########################################################
        collect_t0 = time.time()
        modes = []
        collect_has_stickers = []

        for _ in range(train_env.num_envs):
            collect_has_sticker = np_rng.random() < 0.5
            # PAPERSKIP: Use epsilon-greedy or random noise
            collect_is_random = np_rng.random() < 0.5

            if collect_is_random:
                mode = "random"
            else:
                mode = "train"

            print(f"[Collect] mode: {mode}, has_sticker: {collect_has_sticker}")

            modes.append(mode)
            collect_has_stickers.append(collect_has_sticker)

        train_episodes, train_returns, train_free_energies = fe_run(
            conf,
            np_rng,
            train_env,
            model,
            modes=modes,
            has_stickers=collect_has_stickers,
            sticker_xpos=None,
            calculate_free_energy=False,
        )
        collect_t1 = time.time()
        collect_time = collect_t1 - collect_t0
        print(f"[Collect] Time: {collect_time:.2f}s")

        ############################################################
        # After finish testing model & collecting data, log metrics
        ############################################################
        tot_episodes += train_env.num_envs
        for train_episode in train_episodes:
            episode_store.add_episode_dict(train_episode)

        ########################################################
        # Evaluate current model
        ########################################################
        eval_time = 0
        if (
            conf.eval_period > 0
            and current_step > 0
            and current_step % conf.eval_period == 0
        ):
            should_log_video = (
                conf.record_eval_video_period > 0
                and current_step > 0
                and current_step % conf.record_eval_video_period == 0
            )
            eval_time, _, _ = run_eval_round(
                conf,
                np_rng,
                eval_env,
                model,
                writer,
                eval_return_csv,
                eval_free_energy_csv,
                current_step,
                tot_episodes,
                should_log_video=should_log_video,
            )

        writer.flush()
        eval_return_csv.flush_data()
        eval_free_energy_csv.flush_data()
        episode_t1 = time.time()
        episode_time = episode_t1 - episode_t0
        print(
            f"[Episode] Total Time: {episode_time:.2f}s (Trained steps: {current_step})"
        )
        episode_t0 = episode_t1

        writer.add_scalar(
            "99_system/01_episode_time", episode_time, global_step=tot_episodes
        )
        writer.add_scalar(
            "99_system/02_train_time", train_time, global_step=tot_episodes
        )
        writer.add_scalar(
            "99_system/03_collect_time", collect_time, global_step=tot_episodes
        )
        writer.add_scalar("99_system/04_eval_time", eval_time, global_step=tot_episodes)


def execute():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Log related
    parser.add_argument("--nolog", action="store_true")
    parser.add_argument("--exp_name", default="")

    # Auto load pre-configured configs
    parser.add_argument("--config", nargs="+", default=["robot_mirror"])

    # Other settings
    parser.add_argument("--device", default="-1")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--checkpoint_episode",
        default=None,
        type=int,
        help="Eval only: load checkpoint for this episode (e.g. 42 -> checkpoint_00042.pt). If not set, use latest.",
    )

    args2 = parser.parse_args()
    args_dict = vars(args2)

    # Setup device
    if args2.device == "-1":
        args2.device = get_vacant_gpu(metric="mem_usage")
    args2.device = f"cuda:{args2.device}"
    torch.cuda.set_device(args2.device)

    # Setup seed
    random.seed(args2.seed)
    os.environ["PYTHONHASHSEED"] = str(args2.seed)
    np_rng = np.random.default_rng(args2.seed)
    np.random.seed(args2.seed)  # Torch may still use NumPy's global generator in some paths.
    torch.manual_seed(args2.seed)  # Distribution modules are difficult to wire with custom generators.
    torch.cuda.manual_seed(args2.seed)
    torch.cuda.manual_seed_all(args2.seed)

    # Merge base config and user input config
    with open("configs.yaml", "r") as f:
        configs = YAML(typ="safe").load(f)
        for config_name in ["base", *args2.config]:
            args_dict |= configs[config_name]

    args_dict["log_path"] = None
    args_dict["obs_embed_size"] = None
    args_dict["proprio_size"] = None
    args_dict["action_size"] = None
    conf = ModelConfig(**args_dict)

    use_perfect_determinism = conf.determinism
    if use_perfect_determinism:
        # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
        torch.backends.cuda.matmul.allow_tf32 = True  # MatMul
        torch.backends.cudnn.allow_tf32 = True  # Convolution

    # Use default exp name if not provided
    if conf.exp_name == "":
        conf.exp_name = "/".join(conf.config)

    # Setup log
    print(f"Start {conf.exp_name} (seed {conf.seed}-{args2.label})")
    log_path = f"{conf.logdir}/{conf.exp_name}/{conf.seed}-{args2.label}"
    conf.log_path = log_path
    is_resume = conf.init_from == "resume"

    # Setup config and model from checkpoint when resuming
    if is_resume:
        conf, model, np_rng, checkpoint = load_config_and_model(
            log_path,
            checkpoint_episode=args2.checkpoint_episode,
            device=conf.device,
            seed=args2.seed,
        )
        current_step = checkpoint.get("current_step", 0)
        tot_episodes = checkpoint.get("tot_episodes", 0)
    else:
        checkpoint = None
        current_step = 0
        tot_episodes = 0

    if conf.is_debug:
        shutil.rmtree(log_path, ignore_errors=True)

    if conf.nolog:
        writer = SummaryWriter(write_to_disk=False)  # Dummy~
        eval_return_csv = CSVData(write_to_disk=False)
        eval_free_energy_csv = CSVData(write_to_disk=False)
        hdf5converter = ConvertCSVtoHDF5(write_to_disk=False)
    else:
        os.makedirs(log_path, exist_ok=True)
        os.chmod(log_path, 0o777)
        writer = SummaryWriter(logdir=log_path)
        eval_return_csv = CSVData(log_path, "eval_return", append=is_resume)
        eval_free_energy_csv = CSVData(log_path, "eval_free_energy", append=is_resume)
        csv_data_list = [eval_return_csv, eval_free_energy_csv]
        hdf5converter = ConvertCSVtoHDF5(f"{log_path}/eval.h5", csv_data_list)

    # Setup type
    nptype = np.float32
    torch.set_default_dtype(torch.float32)

    # Create env(s): eval_env + train_env
    envs = {}
    for name, num_envs in [
        ("eval", conf.eval_num_envs),
        ("train", conf.train_num_envs),
    ]:
        env_fns = [lambda i=i: make_env(conf, vector_id=i) for i in range(num_envs)]
        envs[name] = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns, context="spawn")
    eval_env = envs["eval"]
    train_env = envs["train"]

    options = dict(init_random_steps=conf.init_random_steps)
    _ = eval_env.reset(seed=conf.seed, options=options)
    conf.action_size = eval_env.single_action_space.shape[0]
    conf.proprio_size = eval_env.single_observation_space["proprioception"].shape[0]
    _ = train_env.reset(seed=conf.seed, options=options)

    if not conf.nolog:
        with open(f"{log_path}/config.json", "w") as fp:
            json.dump(asdict(conf), fp, indent=4)

    if not is_resume:
        model = Agent(conf=conf, np_rng=np_rng)
        model.to(conf.device, non_blocking=True)

    try:
        # First record initial model state
        checkpoint = {
            "conf": model.conf,
            "model": model.state_dict(),
            "current_step": current_step,
            "tot_episodes": tot_episodes,
            "np_rng_state": np_rng.bit_generator.state,
        }
        if not conf.nolog:
            ckpt_path_num = f"{conf.log_path}/checkpoint_{0:05d}.pt"
            ckpt_path_latest = f"{conf.log_path}/checkpoint.pt"
            print(f"Saving model to {ckpt_path_num}")
            torch.save(checkpoint, ckpt_path_num)
            shutil.copy(ckpt_path_num, ckpt_path_latest)

        run_training(
            conf,
            np_rng,
            train_env,
            eval_env,
            model,
            writer,
            eval_return_csv,
            eval_free_energy_csv,
            current_step,
            tot_episodes,
            checkpoint,
            is_resume,
        )
    finally:
        print("Clean up the data")
        try:
            writer.close()
            eval_return_csv.cleanup()
            eval_free_energy_csv.cleanup()
            hdf5converter.convert()
        except Exception as e:
            print(f"[WARN] Cleanup failed: {e}")
    print("done")


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    torch.set_printoptions(sci_mode=False)
    execute()
