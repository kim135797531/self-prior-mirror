import time
import gymnasium
import torch
import numpy as np
from typing import List
from collections import deque, defaultdict
from einops import rearrange, repeat

from program.models.agent import Agent
from program.models.configs.model_config import ModelConfig
from program.utils import flatten_state, get_scaled_rgb_array_inplace, log_metrics, log_video


@torch.no_grad()
def fe_run(
    conf: ModelConfig,
    np_rng: np.random.Generator,
    env: gymnasium.vector.AsyncVectorEnv,
    agent: Agent,
    modes: List[str],
    has_stickers: List[bool],
    sticker_xpos: int or None,
    calculate_free_energy: bool,
    imitation_queue: int or None = None,
):
    agent.eval()
    device = conf.device

    num_envs = env.num_envs
    assert len(modes) == num_envs and len(has_stickers) == num_envs

    ########################################################
    # Pseudocode line 12~14: Reset env & Initialize data
    ########################################################
    # Pseudocode line 13
    # Reset the environment
    episodes = []
    for mode, has_sticker in zip(modes, has_stickers):
        episode = dict(
            obs_vision=[],
            obs_proprio=[],
            act=[],
            done=[],
            world_vision=[],
            qpos=[],
            sticker_site_xpos=[],
            sticker_site_xmat=[],
            sticker_rgba=[],
            sticker_site_id=[],
            baby_touch_sensor=[],
            baby_touch_sensor2=[],
            is_sticker_detached=[],
            policy_used=False if mode == "random" else True,
            sticker_used=has_sticker,
        )
        episodes.append(episode)

    cur_returns = np.zeros(num_envs, dtype=np.float32)
    cur_free_energies = np.zeros(num_envs, dtype=np.float32)
    done_mask = np.zeros(num_envs, dtype=np.bool_)

    # History: manage only time-axis with deque (each element is a [B, 1, ...] tensor).
    deque_thgpu_post_samples = deque(maxlen=conf.n_imagine_context_steps)
    deque_thgpu_actions = deque(maxlen=conf.n_imagine_context_steps)

    # Pseudocode line 12
    # Reset the environment
    seeds = [int(np_rng.integers(np.iinfo(np.int32).max)) for _ in range(num_envs)]
    option_list = []
    for i in range(num_envs):
        options = dict(
            has_sticker=has_stickers[i],
            sticker_xpos=sticker_xpos,
            imitation_queue=imitation_queue,
            init_random_steps=conf.init_random_steps,
        )
        option_list.append(options)

    # Pseudocode line 14
    # Init new trajectory with the first observation T = {o1}
    next_timesteps, next_infos = env.reset(seed=seeds, options=option_list)

    while not np.all(done_mask):
        th_next_obs_visions = torch.from_numpy(next_timesteps["vision"]).float()
        th_next_obs_visions = get_scaled_rgb_array_inplace(th_next_obs_visions)
        th_next_obs_visions = rearrange(th_next_obs_visions, "b c h w -> b 1 c h w")
        thgpu_next_obs_visions = th_next_obs_visions.to(device, non_blocking=True)

        th_next_obs_proprios = torch.from_numpy(
            next_timesteps["proprioception"]
        ).float()
        th_next_obs_proprios = rearrange(th_next_obs_proprios, "b d -> b 1 d")
        thgpu_next_obs_proprios = th_next_obs_proprios.to(device, non_blocking=True)

        thgpu_next_obs_embeds = agent.obs_provider(
            thgpu_next_obs_visions, thgpu_next_obs_proprios
        )
        _, thgpu_next_post_samples = agent.fe_world.posterior_resample(
            thgpu_next_obs_embeds
        )
        thgpu_next_post_samples = thgpu_next_post_samples.detach()

        np_next_world_visions = next_infos["world_vision"]

        # Pseudocode line 16
        # Infer action a(t) using the action network q(a|s)
        np_actions_random = env.action_space.sample()  # [B, act_dim]
        th_actions_random = torch.from_numpy(np_actions_random)
        th_actions_random = rearrange(th_actions_random, "b d -> b 1 d")
        thgpu_actions_random = th_actions_random.to(device, non_blocking=True)

        if len(deque_thgpu_actions) == 0:
            # Force random action at the first step.
            np_actions = np_actions_random
            th_actions = th_actions_random
            thgpu_actions = thgpu_actions_random
            # Result: both np_actions and th_actions are fully random.
        else:
            thgpu_post_samples = torch.cat(list(deque_thgpu_post_samples), dim=1)
            thgpu_actions = torch.cat(list(deque_thgpu_actions), dim=1)

            batch_size, batch_length = thgpu_post_samples.shape[:2]
            subsequent_mask = torch.ones(
                (batch_size, batch_length, batch_length), device=device
            )
            subsequent_mask = (1 - torch.triu(subsequent_mask, diagonal=1)).bool()
            hiddens = agent.fe_world.transformer(
                thgpu_post_samples, thgpu_actions, subsequent_mask
            )
            last_hiddens = hiddens[:, -1:]
            prior_logits, prior_samples = agent.fe_world.prior_resample(last_hiddens)

            policy_feat = torch.cat([prior_samples, last_hiddens], dim=-1)
            action_dist = agent.efe_policy(policy_feat)
            thgpu_actions = action_dist.rsample().detach()
            th_actions = thgpu_actions.float().cpu()
            np_actions = rearrange(th_actions.numpy(), "b t d -> b (t d)")

            for i in range(num_envs):
                if modes[i] == "random":
                    np_actions[i] = np_actions_random[i]
                    th_actions[i] = th_actions_random[i]
                    thgpu_actions[i] = thgpu_actions_random[i]

        # Pseudocode line 18
        # Add transition to the buffer T += {a(t), o(t+1)} and set t += 1
        for i in range(num_envs):
            episode = episodes[i]
            episode["obs_vision"].append(
                rearrange(th_next_obs_visions[i], "... -> 1 ...")
            )
            episode["obs_proprio"].append(
                rearrange(th_next_obs_proprios[i], "... -> 1 ...")
            )
            episode["act"].append(rearrange(th_actions[i], "... -> 1 ..."))
            episode["done"].append(done_mask[i].reshape(1))
            episode["world_vision"].append(
                rearrange(np_next_world_visions[i], "... -> 1 ...")
            )
            if "qpos" in next_infos:
                episode["qpos"].append(next_infos["qpos"][i].copy())
            if "sticker_site_xpos" in next_infos:
                episode["sticker_site_xpos"].append(
                    next_infos["sticker_site_xpos"][i].copy()
                )
            if "sticker_site_xmat" in next_infos:
                episode["sticker_site_xmat"].append(
                    next_infos["sticker_site_xmat"][i].copy()
                )
            if "sticker_rgba" in next_infos:
                episode["sticker_rgba"].append(next_infos["sticker_rgba"][i].copy())
            if "sticker_site_id" in next_infos:
                episode["sticker_site_id"].append(next_infos["sticker_site_id"][i])
            if "baby_touch_sensor" in next_infos:
                episode["baby_touch_sensor"].append(next_infos["baby_touch_sensor"][i])
            if "baby_touch_sensor2" in next_infos:
                episode["baby_touch_sensor2"].append(next_infos["baby_touch_sensor2"][i])
            if "is_sticker_detached" in next_infos:
                episode["is_sticker_detached"].append(next_infos["is_sticker_detached"][i])

        deque_thgpu_post_samples.append(thgpu_next_post_samples)
        deque_thgpu_actions.append(thgpu_actions)

        # Reward-related computation is usually done in the environment.
        # However, free-energy can be computed from observations alone.
        # Appending here keeps timestep alignment in episode buffers.
        #
        #  RL:
        # (reward_t0 corresponds to obs_t0 and reward obtainable after action_t0)
        #  while True:
        #    action_t0 = agent(obs_t0)
        #    obs_t1, reward_t0 = env.step(action_t0)
        #
        #  FEP:
        # (reward_t0 is the reward associated with state obs_t0)
        #  while True:
        #    action_t0 = agent(obs_t0)
        #    reward_t0 = free_energy(obs_t0)
        #    obs_t1 = env.step(action_t0)
        #
        if calculate_free_energy:
            with torch.no_grad():
                thgpu_free_energies, _ = agent.get_expected_free_energy(
                    thgpu_next_post_samples
                )
                free_energies = thgpu_free_energies.detach().cpu().numpy()
                free_energies = rearrange(free_energies, "b 1 -> b")
        else:
            free_energies = 0
        cur_free_energies += free_energies

        """
        Pseudocode line 17~19
        * Act on the environment with a(t), and receive observation o(t+1)
        * Add transition to the buffer T += {a(t), o(t+1)} and set t += 1
        * Infer state s(t) using the world model
        """
        # Pseudocode line 17
        # Act on the environment with a(t), and receive observation o(t+1)
        next_timesteps, np_rews, terms, truncates, next_infos = env.step(np_actions)
        cur_returns += np_rews

        done_mask = np.logical_or(terms, truncates)

    cur_returns /= conf.timelimit

    return episodes, cur_returns, cur_free_energies

@torch.no_grad()
def run_eval_round(
    conf,
    np_rng,
    eval_env,
    model,
    writer,
    eval_return_csv,
    eval_free_energy_csv,
    current_step,
    tot_episodes,
    should_log_video=False,
):
    """Run one eval round: fe_run, log metrics, append CSV, optional video."""
    print(f"[Eval] Start {conf.eval_num_envs} evals", end="", flush=True)
    eval_t0 = time.time()
    eval_episodes, eval_returns, eval_free_energies = fe_run(
        conf,
        np_rng,
        eval_env,
        model,
        modes=["eval"] * conf.eval_num_envs,
        has_stickers=[True] * conf.eval_num_envs,
        sticker_xpos=None,
        calculate_free_energy=True,
    )
    print("")
    eval_t1 = time.time()
    eval_time = eval_t1 - eval_t0
    print(f"[Eval] Time: {eval_time:.2f}s")

    log_metrics(
        writer,
        is_eval=True,
        tot_episodes=tot_episodes,
        current_step=current_step,
        cur_return=np.average(eval_returns),
        cur_free_energy=np.average(eval_free_energies),
    )
    eval_return_csv.append_data_value(eval_returns)
    eval_free_energy_csv.append_data_value(eval_free_energies)

    if should_log_video and len(eval_episodes) > 0:
        tag = "12_render_eval/episode"
        img = np.concatenate(eval_episodes[0]["obs_vision"], axis=1) + 0.5
        log_video(writer, np_rng, tag, current_step, img, max_render_steps=1000)

    return eval_time, eval_episodes, eval_free_energies