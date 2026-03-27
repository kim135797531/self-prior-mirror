# -*- coding: utf-8 -*-
"""Eval-only entry. Standalone: python 02eval.py log_path [options]. Uses program.fe_eval.run_eval_only."""
import os
import argparse
import torch
import gymnasium.vector
from tensorboardX import SummaryWriter

from program.fe_eval import run_eval_only
from program.envs.env_loader import make_env
from program.load_checkpoint import load_config_and_model
from script.csv_data import CSVData
from script.vacant_gpu import get_vacant_gpu


def main():
    parser = argparse.ArgumentParser(
        description="Run eval-only on a saved run (no 01train.py).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "log_path",
        type=str,
        help="Log directory containing checkpoint (e.g. log/exp_name/42-label).",
    )
    parser.add_argument(
        "--checkpoint_episode",
        type=int,
        default=None,
        help="Load this episode checkpoint (e.g. 42 -> checkpoint_00042.pt). Default: latest.",
    )
    parser.add_argument(
        "--device",
        default="-1",
        help="GPU index or -1 for auto.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--nolog",
        action="store_true",
        help="Dummy writer/CSV (still writes frames/npy to log_path).",
    )
    args = parser.parse_args()

    if args.device == "-1":
        args.device = get_vacant_gpu(metric="mem_usage")
    device = f"cuda:{args.device}"
    torch.cuda.set_device(device)

    conf, model, np_rng, checkpoint = load_config_and_model(
        args.log_path,
        checkpoint_episode=args.checkpoint_episode,
        device=device,
        seed=args.seed,
    )
    conf.log_path = args.log_path
    current_step = checkpoint.get("current_step", 0)
    tot_episodes = checkpoint.get("tot_episodes", 0)
    del checkpoint

    if args.nolog:
        writer = SummaryWriter(write_to_disk=False)
        eval_return_csv = CSVData(write_to_disk=False)
        eval_free_energy_csv = CSVData(write_to_disk=False)
    else:
        os.makedirs(args.log_path, exist_ok=True)
        writer = SummaryWriter(logdir=args.log_path)
        csv_suffix = str(tot_episodes)
        eval_return_csv = CSVData(
            args.log_path, f"eval_return_{csv_suffix}", append=False
        )
        eval_free_energy_csv = CSVData(
            args.log_path, f"eval_free_energy_{csv_suffix}", append=False
        )

    env_fns = [
        lambda i=i: make_env(conf, vector_id=i) for i in range(conf.eval_num_envs)
    ]
    eval_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns, context="spawn")
    options = dict(init_random_steps=conf.init_random_steps)
    _ = eval_env.reset(seed=conf.seed, options=options)
    conf.action_size = eval_env.single_action_space.shape[0]
    conf.proprio_size = eval_env.single_observation_space["proprioception"].shape[0]

    run_eval_only(
        conf,
        np_rng,
        eval_env,
        model,
        writer,
        eval_return_csv,
        eval_free_energy_csv,
        current_step,
        tot_episodes,
    )
    if not args.nolog:
        eval_return_csv.cleanup()
        eval_free_energy_csv.cleanup()
    eval_env.close()
    writer.close()
    print("02eval done.")


if __name__ == "__main__":
    main()
