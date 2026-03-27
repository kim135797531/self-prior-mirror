"""Load config and model from a checkpoint. Shared by main (eval/resume) and replay_episode."""
import glob
import os
import re
import random
import numpy as np
import torch
from dataclasses import asdict

from program.models.agent import Agent
from program.models.configs.model_config import ModelConfig


def get_resume_checkpoint_path(log_path: str) -> tuple[str, int]:
    """Return (checkpoint path, episode number). Uses checkpoint.pt or latest checkpoint_*.pt."""
    ckpt_pt = f"{log_path}/checkpoint.pt"
    if os.path.isfile(ckpt_pt):
        return ckpt_pt, -1
    pattern = f"{log_path}/checkpoint_*.pt"
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found: {ckpt_pt} or {pattern}")

    def episode_num(p):
        m = re.search(r"checkpoint_(\d+)\.pt$", p)
        return int(m.group(1)) if m else -1

    found_path = max(candidates, key=episode_num)
    num = episode_num(found_path)
    print(f"Found checkpoint: {found_path} (episode {num})")
    return found_path, num


def load_config_and_model(
    log_path: str,
    checkpoint_episode=None,
    device=None,
    seed=42,
):
    """
    Load conf and model from checkpoint under log_path.
    Returns (conf, model, np_rng, checkpoint).
    checkpoint contains current_step, tot_episodes, model state, and for resume also optimizers etc.
    """
    random.seed(seed)
    np_rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if checkpoint_episode is not None:
        ckpt_path = f"{log_path}/checkpoint_{checkpoint_episode:05d}.pt"
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"Using checkpoint: {ckpt_path}")
    else:
        ckpt_path, checkpoint_episode = get_resume_checkpoint_path(log_path)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    checkpoint_conf = checkpoint["conf"]
    checkpoint_conf.checkpoint_episode = checkpoint_episode
    conf_dict = asdict(checkpoint_conf)
    conf_dict["log_path"] = log_path
    if device is not None:
        conf_dict["device"] = device
    conf = ModelConfig(**conf_dict)

    if "np_rng_state" in checkpoint:
        np_rng.bit_generator.state = checkpoint["np_rng_state"]

    model = Agent(conf=conf, np_rng=np_rng)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(conf.device, non_blocking=True)
    return conf, model, np_rng, checkpoint
