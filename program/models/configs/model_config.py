#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""model_config.py
Created by Dongmin Kim at 24. 8. 9.

This module does stuff.
"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Model Config
    label: int

    """
    Filled with program
    """
    # Basic
    nolog: bool
    exp_name: str
    config: list
    device: str
    seed: int
    log_path: str

    # Coupled with Env
    obs_embed_size: int  # # CNN encoder output size (1024)
    proprio_size: int  # 18 in half-cheetah
    action_size: int  # 6 in half-cheetah

    """
    Defined in yaml file
    """
    # Basic
    is_debug: bool
    suite: str
    init_from: str
    logdir: str
    compile: bool
    determinism: bool
    use_decay_world: bool
    use_decay_policy: bool
    use_decay_self_prior: bool
    warmup_iters: int
    lr_decay_iters: int
    use_zclip_world: bool
    use_zclip_policy: bool
    use_zclip_self_prior: bool
    sticker_detach_distance_threshold: float
    # Train
    save_model_period: int
    recon_model_period: int
    record_train_video_period: int
    checkpoint_save_episode_store: bool
    # Eval
    eval_period: int
    record_eval_video_period: int
    checkpoint_episode: int

    # Env (Unit is "real" env steps)
    action_repeat: int
    init_random_steps: int
    detach_step_threshold: int

    # Env (Unit is "model" steps)
    timelimit: int

    # Runner
    train_num_envs: int
    eval_num_envs: int
    # runner_n_context_steps: int

    # Training setting
    use_amp: bool
    n_epochs: int
    total_steps: int
    max_episodes: int
    max_episode_stores: int
    train_world_episodes_from: int
    train_self_prior_episodes_from: int
    train_policy_episodes_from: int

    # Data sampling
    replay_on_gpu: bool  # PAPERSKIP: Not implemented
    n_paths: int
    n_steps: int
    n_imagine_paths: int
    n_imagine_context_steps: int
    n_imagine_steps: int

    # Model common
    reward_symlog_classes: int
    reward_symlog_lower_bound: float
    reward_symlog_upper_bound: float

    # World model
    cnn_channel_depth: int
    world_stoch_size: int
    world_class_size: int
    world_hidden_size: int
    transformer_max_steps: int
    transformer_n_layers: int
    transformer_n_heads: int
    transformer_dropout: float

    # World training
    world_lr: float
    min_world_lr: float
    world_free_nats: float
    world_dynamics_kl_scale: float
    world_representation_kl_scale: float
    world_grad_clip: float

    # Policy model
    use_reinforce: bool
    policy_hidden_size: int
    value_hidden_size: int
    policy_layers: int

    # Policy training
    policy_lr: float
    min_policy_lr: float
    policy_adam_eps: float
    advantage_ema_decay: float
    advantage_ema_lower_bound: float
    advantage_ema_upper_bound: float
    entropy_temperature: float
    discount_gamma: float
    gae_lambda: float
    update_slow_critic_decay: float
    policy_grad_clip: float

    # Self-prior model
    # Self-prior training
    self_prior_hidden_size: int
    self_prior_n_layers: int
    self_prior_n_heads: int
    self_prior_lr: float
    min_self_prior_lr: float
    self_prior_grad_clip: float
    use_slow_self_prior: bool
    update_slow_self_prior_decay: float
