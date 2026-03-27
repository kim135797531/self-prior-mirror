# -*- coding: utf-8 -*-
"""Shared eval-only logic: run_eval_only = run_eval_round + save_eval_only_artifacts. Used by 01train and 02eval."""
import os
import csv
import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from einops import rearrange
from PIL import Image

from program.fe_runner import run_eval_round
from program.envs.env_loader import make_env


def _save_vision_img(img, path):
    """Save a single vision image to path. Accepts tensor or numpy; (C,H,W) or (1,C,H,W)."""
    if torch.is_tensor(img):
        img = torch.clamp(img + 0.5, 0, 1).cpu().numpy()
    if img.ndim == 4:
        img = img.squeeze(0)
    if img.ndim == 3:
        c, h, w = img.shape
        if c == 1:
            img = img.squeeze(0)
        else:
            img = np.transpose(img, (1, 2, 0))
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_u8).save(path)


def _assert_eval_touch_sensors_nonnegative(episode):
    """Eval inspection: raw logged touch sensors must be non-negative (see env get_info)."""
    for name in ("baby_touch_sensor", "baby_touch_sensor2"):
        for i, v in enumerate(episode.get(name, [])):
            fv = float(v)
            assert fv >= 0.0, f"eval inspection: {name}[{i}]={fv} must be >= 0"


def _save_hand_sticker_distance_csv(conf, eval_episodes, tot_episodes):
    """Save per-step sticker state + sensor values for all eval episodes in one CSV."""
    csv_path = os.path.join(conf.log_path, f"hand_sticker_distance_{tot_episodes}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode_idx",
                "step",
                "baby_touch_sensor",
                "baby_touch_sensor2",
                "hand_sticker_distance",
                "is_sticker_detached",
                "sticker_site_id",
                "sticker_site_xpos_x",
                "sticker_site_xpos_y",
                "sticker_site_xpos_z",
                "sticker_site_xmat_00",
                "sticker_site_xmat_01",
                "sticker_site_xmat_02",
                "sticker_site_xmat_10",
                "sticker_site_xmat_11",
                "sticker_site_xmat_12",
                "sticker_site_xmat_20",
                "sticker_site_xmat_21",
                "sticker_site_xmat_22",
                "sticker_rgba_r",
                "sticker_rgba_g",
                "sticker_rgba_b",
                "sticker_rgba_a",
            ]
        )
        for episode_idx, episode in enumerate(eval_episodes):
            if (
                len(episode["baby_touch_sensor"]) == 0
                or len(episode["baby_touch_sensor2"]) == 0
            ):
                print(
                    f"Skip hand-sticker CSV episode {episode_idx}: required sensor values are missing."
                )
                continue

            distances = _compute_episode_hand_sticker_distances(episode)
            detached_flags = _compute_episode_sticker_detached_flags(episode)
            for step, (
                right_sensor_value,
                left_sensor_value,
                distance,
                is_sticker_detached,
            ) in enumerate(
                zip(
                    episode["baby_touch_sensor"],
                    episode["baby_touch_sensor2"],
                    distances,
                    detached_flags,
                )
            ):
                sticker_site_id = int(
                    episode.get("sticker_site_id", [-1] * len(distances))[step]
                )
                sticker_site_xpos = np.asarray(
                    episode.get("sticker_site_xpos", [np.zeros(3)] * len(distances))[step],
                    dtype=float,
                ).reshape(-1)
                sticker_site_xmat = np.asarray(
                    episode.get("sticker_site_xmat", [np.zeros(9)] * len(distances))[step],
                    dtype=float,
                ).reshape(-1)
                sticker_rgba = np.asarray(
                    episode.get("sticker_rgba", [np.zeros(4)] * len(distances))[step],
                    dtype=float,
                ).reshape(-1)
                writer.writerow(
                    [
                        episode_idx,
                        step,
                        right_sensor_value,
                        left_sensor_value,
                        distance,
                        int(is_sticker_detached),
                        sticker_site_id,
                        float(sticker_site_xpos[0]) if sticker_site_xpos.size > 0 else 0.0,
                        float(sticker_site_xpos[1]) if sticker_site_xpos.size > 1 else 0.0,
                        float(sticker_site_xpos[2]) if sticker_site_xpos.size > 2 else 0.0,
                        float(sticker_site_xmat[0]) if sticker_site_xmat.size > 0 else 0.0,
                        float(sticker_site_xmat[1]) if sticker_site_xmat.size > 1 else 0.0,
                        float(sticker_site_xmat[2]) if sticker_site_xmat.size > 2 else 0.0,
                        float(sticker_site_xmat[3]) if sticker_site_xmat.size > 3 else 0.0,
                        float(sticker_site_xmat[4]) if sticker_site_xmat.size > 4 else 0.0,
                        float(sticker_site_xmat[5]) if sticker_site_xmat.size > 5 else 0.0,
                        float(sticker_site_xmat[6]) if sticker_site_xmat.size > 6 else 0.0,
                        float(sticker_site_xmat[7]) if sticker_site_xmat.size > 7 else 0.0,
                        float(sticker_site_xmat[8]) if sticker_site_xmat.size > 8 else 0.0,
                        float(sticker_rgba[0]) if sticker_rgba.size > 0 else 0.0,
                        float(sticker_rgba[1]) if sticker_rgba.size > 1 else 0.0,
                        float(sticker_rgba[2]) if sticker_rgba.size > 2 else 0.0,
                        float(sticker_rgba[3]) if sticker_rgba.size > 3 else 0.0,
                    ]
                )
    print(f"Saved hand-sticker distance CSV to {csv_path}")


def _compute_episode_sticker_detached_flags(episode):
    """Return per-step detach flags saved from env info."""
    return [bool(value) for value in episode.get("is_sticker_detached", [])]


def _compute_episode_hand_sticker_distances(episode):
    """Return per-step nearest hand-to-sticker distances for one eval episode."""
    right_values = episode.get("baby_touch_sensor", [])
    left_values = episode.get("baby_touch_sensor2", [])
    if right_values and left_values:
        _assert_eval_touch_sensors_nonnegative(episode)
    detached_flags = _compute_episode_sticker_detached_flags(episode)
    distances = []
    for step, (right_sensor_value, left_sensor_value) in enumerate(
        zip(right_values, left_values)
    ):
        if step < len(detached_flags) and detached_flags[step]:
            distances.append(0.0)
            continue
        distance = min(float(right_sensor_value), float(left_sensor_value))
        distances.append(max(distance, 0.0))
    return distances


def _save_eval_episode_summary_csv(conf, eval_episodes, tot_episodes, eval_free_energies):
    """Save one summary row per eval episode for downstream plotting."""
    csv_path = os.path.join(conf.log_path, f"eval_episode_summary_{tot_episodes}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode_idx",
                "num_steps",
                "total_efe",
                "min_hand_sticker_distance",
                "avg_hand_sticker_distance",
                "sticker_detached",
            ]
        )

        for episode_idx, (episode, total_efe) in enumerate(
            zip(eval_episodes, np.asarray(eval_free_energies).tolist())
        ):
            distances = _compute_episode_hand_sticker_distances(episode)
            detached_flags = _compute_episode_sticker_detached_flags(episode)
            if distances:
                min_distance = float(np.min(distances))
                avg_distance = float(np.mean(distances))
                sticker_detached = any(detached_flags)
            else:
                min_distance = float("nan")
                avg_distance = float("nan")
                sticker_detached = False

            writer.writerow(
                [
                    episode_idx,
                    len(episode.get("obs_vision", [])),
                    float(total_efe),
                    min_distance,
                    avg_distance,
                    int(sticker_detached),
                ]
            )
    print(f"Saved eval episode summary CSV to {csv_path}")


def _save_eval_episode_replay_arrays(conf, episode, tot_episodes, episode_idx):
    """Save replay arrays for one eval episode."""
    np.save(
        os.path.join(conf.log_path, f"eval_episode_{episode_idx}_qpos_{tot_episodes}.npy"),
        np.stack([np.asarray(p).flatten() for p in episode["qpos"]]),
    )


def _save_proprio_array(proprio, path):
    """Save reconstructed proprio as (N, D) numpy array."""
    if torch.is_tensor(proprio):
        proprio = proprio.detach().cpu().numpy()
    proprio = np.asarray(proprio)
    if proprio.ndim == 3 and proprio.shape[1] == 1:
        proprio = proprio.squeeze(1)
    np.save(path, proprio)


def _save_render_from_proprio_artifacts(conf, episode, proprio_recons, save_dir):
    """Render camera images from reconstructed proprio on top of replay base states."""
    render_env = make_env(conf, vector_id=0)
    render_unwrapped = render_env.unwrapped
    if not hasattr(render_unwrapped, "render_from_reconstructed_proprio"):
        print("Skip proprio rerender artifacts: env does not support helper.")
        render_env.close()
        return

    seed = getattr(conf, "seed", None)
    reset_options = dict(init_random_steps=conf.init_random_steps)

    try:
        render_env.reset(seed=seed, options=reset_options)

        for key, proprio_recon in proprio_recons.items():
            _save_proprio_array(
                proprio_recon,
                os.path.join(save_dir, f"recon_proprio_{key}.npy"),
            )

        n_available = next(iter(proprio_recons.values())).shape[0]
        for i in range(n_available):
            qpos = np.asarray(episode["qpos"][i]).flatten()
            sticker_site_xpos = np.asarray(episode["sticker_site_xpos"][i]).flatten()
            sticker_site_xmat = np.asarray(episode["sticker_site_xmat"][i]).flatten()
            sticker_rgba = np.asarray(episode["sticker_rgba"][i]).flatten()

            for key, proprio_recon in proprio_recons.items():
                proprio_i = proprio_recon[i].reshape(-1).detach().cpu().numpy()
                obs, info = render_unwrapped.render_from_reconstructed_proprio(
                    qpos=qpos,
                    sticker_site_xpos=sticker_site_xpos,
                    sticker_site_xmat=sticker_site_xmat,
                    sticker_rgba=sticker_rgba,
                    proprio_normalized=proprio_i,
                )

                # Direct save since unwrapped env has original RGB array
                img = np.transpose(obs["vision"], (1, 2, 0))
                Image.fromarray(img).save(
                    os.path.join(save_dir, f"render_from_proprio_{key}_{i:02d}.png")
                )
                img = np.transpose(info["world_vision"], (1, 2, 0))
                Image.fromarray(img).save(
                    os.path.join(
                        save_dir, f"world_render_from_proprio_{key}_{i:02d}.png"
                    )
                )
    finally:
        render_env.close()


@torch.no_grad()
def save_eval_only_artifacts(
    conf,
    model,
    eval_episodes,
    eval_free_energies,
    tot_episodes,
    writer,
    eval_return_csv,
    eval_free_energy_csv,
):
    """Save frames, proprio, qpos, sticker, self-prior samples, multimodal restoration; flush writer/CSV."""

    save_dir_frames = os.path.join(
        conf.log_path, f"eval_episode_0_frames_{tot_episodes}"
    )
    os.makedirs(save_dir_frames, exist_ok=True)

    # for ep_idx, ep in enumerate(eval_episodes):
    ep = eval_episodes[0]
    ep_idx = 0
    print("Save all obs_vision frames of the first eval episode as PNGs")
    for t, frame in enumerate(ep["obs_vision"]):
        _save_vision_img(
            frame.squeeze(0),
            os.path.join(save_dir_frames, f"episode_{ep_idx:02d}_frame_{t:05d}.png"),
        )

    ep0 = eval_episodes[0]

    # Save replay arrays and one consolidated hand-sticker CSV for all eval episodes
    for ep_idx, ep in enumerate(eval_episodes):
        _save_eval_episode_replay_arrays(conf, ep, tot_episodes, episode_idx=ep_idx)
    _save_hand_sticker_distance_csv(conf, eval_episodes, tot_episodes)
    _save_eval_episode_summary_csv(conf, eval_episodes, tot_episodes, eval_free_energies)

    print("d-kim: Self-prior: sample 10 images and save as JPG")
    n_samples = 10
    self_prior_samples = model.self_prior.get_sample(n_samples)
    self_prior_vision, _ = model.obs_provider.decode_from_feature(self_prior_samples)
    save_dir = os.path.join(conf.log_path, f"self_prior_{tot_episodes}")
    os.makedirs(save_dir, exist_ok=True)
    for i in range(n_samples):
        _save_vision_img(self_prior_vision[i], f"{save_dir}/sample_{i:02d}.png")
    print(f"Saved {n_samples} self-prior samples to {save_dir}")

    print(
        "d-kim: Multimodal restoration: vision-only latent -> decoder-only vs self-prior refined"
    )
    n_examples = 10
    n_available = min(n_examples, len(ep0["obs_vision"]))
    obs_vision = (
        torch.stack([ep0["obs_vision"][k].squeeze(0) for k in range(n_available)])
        .float()
        .to(conf.device, non_blocking=True)
    )
    obs_proprio = (
        torch.stack([ep0["obs_proprio"][k].squeeze(0) for k in range(n_available)])
        .float()
        .to(conf.device, non_blocking=True)
    )
    embed_v = model.obs_provider(obs_vision, torch.zeros_like(obs_proprio))
    embed_p = model.obs_provider(torch.zeros_like(obs_vision), obs_proprio)
    _, feat_v = model.fe_world.posterior_resample(embed_v)
    _, feat_p = model.fe_world.posterior_resample(embed_p)
    vision_recon_v, proprio_recon_v = model.obs_provider.decode_from_feature(feat_v)
    vision_recon_p, proprio_recon_p = model.obs_provider.decode_from_feature(feat_p)

    def recon_from_feat(feat):
        bos_x = model.self_prior.prepare(feat)
        logits = model.self_prior.forward(bos_x[:, :-1])
        bos_idx = model.self_prior.num_classes
        logits[:, :, bos_idx] = -1e9
        sampled = Categorical(logits=logits).sample()
        num_classes = model.self_prior.num_classes
        self_feat = F.one_hot(sampled, num_classes=num_classes).float()
        self_feat = rearrange(self_feat, "B K C -> B (K C)")
        self_feat = self_feat.unsqueeze(1)
        vision_recon_self, proprio_recon_self = model.obs_provider.decode_from_feature(
            self_feat
        )
        return vision_recon_self, proprio_recon_self

    vision_recon_self_v, proprio_recon_self_v = recon_from_feat(feat_v)
    vision_recon_self_p, proprio_recon_self_p = recon_from_feat(feat_p)

    save_dir_mm = os.path.join(conf.log_path, f"multimodal_restoration_{tot_episodes}")
    os.makedirs(save_dir_mm, exist_ok=True)
    for i in range(n_available):
        _save_vision_img(obs_vision[i], f"{save_dir_mm}/input_vision_{i:02d}.png")
        _save_vision_img(vision_recon_v[i], f"{save_dir_mm}/vision_recon_v_{i:02d}.png")
        _save_vision_img(vision_recon_p[i], f"{save_dir_mm}/vision_recon_p_{i:02d}.png")
        _save_vision_img(
            vision_recon_self_v[i], f"{save_dir_mm}/vision_recon_self_v_{i:02d}.png"
        )
        _save_vision_img(
            vision_recon_self_p[i], f"{save_dir_mm}/vision_recon_self_p_{i:02d}.png"
        )

    _save_render_from_proprio_artifacts(
        conf,
        ep0,
        {
            "v": proprio_recon_v,
            "p": proprio_recon_p,
            "self_v": proprio_recon_self_v,
            "self_p": proprio_recon_self_p,
        },
        save_dir_mm,
    )
    print(f"Saved {n_available} multimodal restoration images to {save_dir_mm}")

    writer.flush()
    eval_return_csv.flush_data()
    eval_free_energy_csv.flush_data()


@torch.no_grad()
def run_eval_only(
    conf,
    np_rng,
    eval_env,
    model,
    writer,
    eval_return_csv,
    eval_free_energy_csv,
    current_step,
    tot_episodes,
):
    """One eval round (fe_run, log_metrics, optional video) then save_eval_only_artifacts. Shared by 01train and 02eval."""
    model.eval()
    print("Run eval only for eval_rounds rounds")
    _, eval_episodes, eval_free_energies = run_eval_round(
        conf,
        np_rng,
        eval_env,
        model,
        writer,
        eval_return_csv,
        eval_free_energy_csv,
        current_step,
        tot_episodes,
        should_log_video=conf.record_eval_video_period > 0,
    )
    save_eval_only_artifacts(
        conf,
        model,
        eval_episodes,
        eval_free_energies,
        tot_episodes,
        writer,
        eval_return_csv,
        eval_free_energy_csv,
    )
