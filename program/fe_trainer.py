import math
import itertools
import numpy as np
import torch
from einops import rearrange
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, _warn_get_lr_called_within_step

from program.models.agent import Agent
from program.models.configs.model_config import ModelConfig
from program.utils import (
    FreezeParameters,
    RunningMeanStd,
    storm_calc_lambda_return,
    sheeprl_compute_lambda_values,
    SymLogTwoHotLoss,
    EMAScalar,
    percentile,
    mse_loss_func,
)
from program.zclip import ZClip


class WarmupCosineLRScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        use_decay: bool,
        warmup_iters: int,
        lr_decay_iters: int,
        learning_rate: float,
        min_learning_rate: float = None,
        last_epoch: int = -1,  # Use -1 to start from the beginning
    ):
        self.optimizer = optimizer

        # Turn on/off lr decay
        self.use_decay = use_decay

        # 0 ~ warmup: Linear increase
        self.warmup_iters = warmup_iters

        # warmup ~ decay: Cosine decay
        self.lr_decay_iters = lr_decay_iters

        self.lr = learning_rate
        self.min_lr = min_learning_rate

        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    # From nanoGPT
    def get_lr(self):
        _warn_get_lr_called_within_step(self)

        if self.use_decay is not True:
            final_lr = self.lr
        elif self.last_epoch < self.warmup_iters:
            # 1) linear warmup for warmup_iters steps
            final_lr = self.lr * ((self.last_epoch + 1) / (self.warmup_iters + 1))
        elif self.last_epoch > self.lr_decay_iters:
            # 2) if it > lr_decay_iters, return min learning rate
            final_lr = self.min_lr
        else:
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (self.last_epoch - self.warmup_iters) / (
                self.lr_decay_iters - self.warmup_iters
            )
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            final_lr = self.min_lr + coeff * (self.lr - self.min_lr)

        return [final_lr for group in self.optimizer.param_groups]


# PAPER: FETrainer is also computationally heavy; consider moving it into Agent so functions can be compiled together.
class FETrainer:
    def __init__(self, conf: ModelConfig, np_rng: np.random.Generator, agent: Agent):
        self.conf = conf
        self.np_rng = np_rng
        self.agent = agent

        # Learning parameters (common)
        self.reward_symlog_classes = conf.reward_symlog_classes
        self.reward_symlog_lower_bound = conf.reward_symlog_lower_bound
        self.reward_symlog_upper_bound = conf.reward_symlog_upper_bound

        # Learning parameters (fep)
        self.world_free_nats = conf.world_free_nats
        self.world_dynamics_kl_scale = conf.world_dynamics_kl_scale
        self.world_representation_kl_scale = conf.world_representation_kl_scale
        self.world_grad_clip = conf.world_grad_clip

        # Learning parameters (efep)
        self.advantage_ema_decay = conf.advantage_ema_decay
        self.advantage_ema_lower_bound = conf.advantage_ema_lower_bound
        self.advantage_ema_upper_bound = conf.advantage_ema_upper_bound
        self.entropy_temperature = conf.entropy_temperature
        self.discount_gamma = conf.discount_gamma
        self.gae_lambda = conf.gae_lambda

        self.ambiguity_rms = RunningMeanStd()
        self.ambiguity_beta = 1e-3
        self.actor_ema_lower_bound = EMAScalar(decay=self.advantage_ema_decay)
        self.actor_ema_upper_bound = EMAScalar(decay=self.advantage_ema_decay)
        self.update_slow_critic_decay = conf.update_slow_critic_decay
        self.policy_grad_clip = conf.policy_grad_clip

        self.self_prior_grad_clip = conf.self_prior_grad_clip

        self.symlog_twohot_loss = SymLogTwoHotLoss(
            conf,
            self.reward_symlog_classes,
            self.reward_symlog_lower_bound,
            self.reward_symlog_upper_bound,
        )

        self.use_zclip_world = conf.use_zclip_world
        self.use_zclip_policy = conf.use_zclip_policy
        self.use_zclip_self_prior = conf.use_zclip_self_prior
        self.world_zclip = ZClip()
        self.policy_zclip = ZClip()
        self.critic_zclip = ZClip()
        self.self_prior_zclip = ZClip()

    def get_checkpoint_state(self):
        """State for resume: EMA/RMS and ZClip used in training."""
        out = {
            "ambiguity_rms": self.ambiguity_rms.state_dict(),
            "actor_ema_lower_bound": self.actor_ema_lower_bound.state_dict(),
            "actor_ema_upper_bound": self.actor_ema_upper_bound.state_dict(),
        }
        out["world_zclip"] = self.world_zclip.state_dict()
        out["policy_zclip"] = self.policy_zclip.state_dict()
        out["critic_zclip"] = self.critic_zclip.state_dict()
        out["self_prior_zclip"] = self.self_prior_zclip.state_dict()
        return out

    def load_checkpoint_state(self, state: dict):
        """Restore EMA/RMS and ZClip from checkpoint."""
        if "ambiguity_rms" in state:
            self.ambiguity_rms.load_state_dict(state["ambiguity_rms"])
        if "actor_ema_lower_bound" in state:
            self.actor_ema_lower_bound.load_state_dict(state["actor_ema_lower_bound"])
        if "actor_ema_upper_bound" in state:
            self.actor_ema_upper_bound.load_state_dict(state["actor_ema_upper_bound"])
        if "world_zclip" in state:
            self.world_zclip.load_state_dict(state["world_zclip"])
        if "policy_zclip" in state:
            self.policy_zclip.load_state_dict(state["policy_zclip"])
        if "critic_zclip" in state:
            self.critic_zclip.load_state_dict(state["critic_zclip"])
        if "self_prior_zclip" in state:
            self.self_prior_zclip.load_state_dict(state["self_prior_zclip"])

    def configure_optimizers(
        self,
        optimizer_name,
        target_modules,
        learning_rate,
        min_learning_rate,
        use_decay,
        eps=1e-8,
    ):
        optim_groups = []
        for target_module in target_modules:
            optim_groups += list(target_module.parameters())

        if optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                optim_groups, lr=learning_rate, eps=eps, fused=True
            )
        elif optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                optim_groups, lr=learning_rate, eps=eps, fused=True
            )
        else:
            raise ValueError(f"Invalid optimizer name: {optimizer_name}")
        scheduler = WarmupCosineLRScheduler(
            optimizer=optimizer,
            use_decay=use_decay,
            warmup_iters=self.conf.warmup_iters,
            lr_decay_iters=self.conf.lr_decay_iters,
            learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
        )
        return optimizer, scheduler

    def train_world(
        self,
        world_scaler,
        world_optimizer,
        world_scheduler: WarmupCosineLRScheduler,
        obs_visions,
        obs_proprios,
        acts,
        get_reconstruction=False,
    ):
        self.agent.train()

        with FreezeParameters(
            [
                self.agent.efe_policy,
                self.agent.efe_value,
                self.agent.efe_value_target,
                self.agent.self_prior,
            ]
        ) and torch.autocast(
            device_type=self.conf.device,
            dtype=torch.bfloat16,
            enabled=self.conf.use_amp,
        ):
            # F optimization.
            # In other words, calculate the variational free energy for the past (dataset).
            # Create loss to optimize variational free energy.
            free_energy_dict = dict()

            batch_size, batch_length = obs_visions.shape[:2]
            device = obs_visions.device

            #
            # Encoding & Decoding
            #
            obs_embeds = self.agent.obs_provider(obs_visions, obs_proprios)
            post_logits, post_samples = self.agent.fe_world.posterior_resample(
                obs_embeds
            )
            obs_vision_recons, obs_proprio_recons = (
                self.agent.obs_provider.decode_from_feature(post_samples)
            )

            #
            # Transformer
            #
            temporal_masks = torch.ones((1, batch_length, batch_length), device=device)
            temporal_masks = (1 - torch.triu(temporal_masks, diagonal=1)).bool()

            hiddens = self.agent.fe_world.transformer(
                post_samples, acts, temporal_masks
            )
            prior_logits, prior_samples = self.agent.fe_world.prior_resample(hiddens)

            #
            # Reconstruction loss
            #
            vision_recon_loss = mse_loss_func(obs_vision_recons, obs_visions)
            proprio_recon_loss = mse_loss_func(obs_proprio_recons, obs_proprios)
            recon_loss = vision_recon_loss + proprio_recon_loss

            #
            # Decoder loss
            #
            # PAPER: This loss is unnecessary when self-prior uses discrete features.
            # STORM also does not include this loss.
            # We may later revisit how to build self-prior directly in discrete space.
            #
            # PAPER: Removing this slightly hurt performance
            #
            detached_obs_embeds = obs_embeds.detach()
            obs_vision_recons_from_detached, obs_proprio_recons_from_detached = (
                self.agent.obs_provider.decode_from_embed(detached_obs_embeds)
            )
            vision_decoder_loss = mse_loss_func(
                obs_vision_recons_from_detached, obs_visions
            )
            proprio_decoder_loss = mse_loss_func(
                obs_proprio_recons_from_detached, obs_proprios
            )
            decoder_loss = vision_decoder_loss + proprio_decoder_loss

            #
            # Dynamic loss
            #
            dynamics_kl_div, dynamics_real_kl_div = (
                self.agent.categorical_kl_div_loss_func(
                    p_logits=post_logits[:, 1:].detach(),
                    q_logits=prior_logits[:, :-1],
                    free_bits=self.world_free_nats,
                )
            )
            representation_kl_div, representation_real_kl_div = (
                self.agent.categorical_kl_div_loss_func(
                    p_logits=post_logits[:, 1:],
                    q_logits=prior_logits[:, :-1].detach(),
                    free_bits=self.world_free_nats,
                )
            )
            kl_loss = (
                self.world_dynamics_kl_scale * dynamics_kl_div
                + self.world_representation_kl_scale * representation_kl_div
            )
            kl_loss = kl_loss.mean()

            dynamics_loss = dynamics_kl_div.mean()
            representation_loss = representation_kl_div.mean()
            dynamics_real_kl_div = dynamics_real_kl_div.mean()
            representation_real_kl_div = representation_real_kl_div.mean()

            # Final loss
            world_loss = recon_loss + decoder_loss + kl_loss
            # world_loss = recon_loss + kl_loss

        world_optimizer.zero_grad()
        world_scaler.scale(world_loss).backward()
        world_scaler.unscale_(world_optimizer)

        world_parameters = itertools.chain(
            self.agent.obs_provider.parameters(),
            self.agent.fe_world.parameters(),
        )
        if self.use_zclip_world:
            grad_norm_world = self.world_zclip.step_with_parameters(world_parameters)
        else:
            grad_norm_world = torch.nn.utils.clip_grad_norm_(
                world_parameters,
                self.world_grad_clip,
            ).item()

        world_scaler.step(world_optimizer)
        world_scaler.update()
        world_optimizer.zero_grad(set_to_none=True)
        world_scheduler.step()

        free_energy_dict["01_world_loss"] = world_loss.item()
        free_energy_dict["02_world_grad_norm"] = grad_norm_world
        free_energy_dict["11_recon_loss"] = recon_loss.item()
        free_energy_dict["12_decoder_loss"] = decoder_loss.item()
        free_energy_dict["21_dynamics_loss"] = dynamics_loss.item()
        free_energy_dict["22_dynamics_real_kl_div"] = dynamics_real_kl_div.mean().item()
        free_energy_dict["23_representation_loss"] = representation_loss.item()
        free_energy_dict["24_representation_real_kl_div"] = (
            representation_real_kl_div.mean().item()
        )

        reconstruction_dict = dict()
        return free_energy_dict, reconstruction_dict

    def train_policy_value(
        self,
        actor_scaler: torch.amp.GradScaler,
        actor_optimizer,
        actor_scheduler,
        critic_scaler,
        critic_optimizer,
        critic_scheduler,
        imagine_policy_features,  # [s0h0], [s1h1], [s2h2], ...
        imagine_actions,  # a(s0h0), a(s1h1), a(s2h2), ...
        imagine_free_energies,  # G(s1), G(s2), ...
    ):
        self.agent.train()

        # Calculate expected free energy (actor loss) and infinite horizon estimation function (critic loss) separately.
        # Then, add them well to calculate the infinite horizon expected free energy.
        policy_dict = dict()

        # For stability, when updating the actor, fix the world model and utility.
        with FreezeParameters(
            [
                self.agent.obs_provider,
                self.agent.fe_world,
                self.agent.self_prior,
                self.agent.efe_value_target,
            ]
        ) and torch.autocast(
            device_type=self.conf.device,
            dtype=torch.bfloat16,
            enabled=self.conf.use_amp,
        ):
            # ==================================================================================== #
            # Start actor loss
            # ==================================================================================== #
            # Predict values, rewards and continues
            qv = self.agent.efe_value(imagine_policy_features)
            predicted_values = self.symlog_twohot_loss.decode(qv)
            prev_predicted_values = predicted_values[:, :-1]
            next_predicted_values = predicted_values[:, 1:]

            # Assuming env never end during episode
            continues = torch.ones(
                imagine_policy_features.shape[:2], device=imagine_policy_features.device
            )

            with torch.no_grad():
                discount = (
                    torch.cumprod(continues * self.discount_gamma, dim=1)
                    / self.discount_gamma
                )

            next_continues = continues[:, 1:]
            lambda_values = sheeprl_compute_lambda_values(
                imagine_free_energies,
                next_predicted_values,
                next_continues * self.discount_gamma,
                self.gae_lambda,
            )

            lower_bound = self.actor_ema_lower_bound(
                percentile(lambda_values, self.advantage_ema_lower_bound).item()
            )
            upper_bound = self.actor_ema_upper_bound(
                percentile(lambda_values, self.advantage_ema_upper_bound).item()
            )
            policy_norm_ratio = upper_bound - lower_bound
            invscale = max(1, policy_norm_ratio)
            disadvantages = (lambda_values - prev_predicted_values) / invscale

            # 1. Calculate expected free energy (actor loss)
            # Predict the action distribution for each step of imagined trajectories.
            actor_dist = self.agent.efe_policy(imagine_policy_features.detach()[:, :-1])
            imagine_entropy = actor_dist.entropy()

            if self.conf.use_reinforce:
                imagine_log_prob = actor_dist.log_prob(imagine_actions.detach())
                neg_objective = imagine_log_prob * disadvantages.detach()
            else:
                neg_objective = disadvantages

            entropy = self.entropy_temperature * imagine_entropy
            entropy_loss = -1 * entropy
            actor_loss = torch.mean(
                discount[:, :-1].detach() * (neg_objective + entropy_loss)
            )

            # ==================================================================================== #
            # Start critic loss
            # ==================================================================================== #

            qv_from_detach = self.agent.efe_value(imagine_policy_features.detach())

            with torch.no_grad():
                self.agent.efe_value_target.eval()
                target_qv = self.agent.efe_value_target(
                    imagine_policy_features.detach()
                )
                predicted_target_values = self.symlog_twohot_loss.decode(target_qv)

            slow_value_regularization_loss = self.symlog_twohot_loss.forward(
                qv_from_detach[:, :-1], predicted_target_values.detach()[:, :-1]
            )
            value_loss = self.symlog_twohot_loss.forward(
                qv_from_detach[:, :-1], lambda_values.detach()
            )

            critic_loss = value_loss + slow_value_regularization_loss
            critic_loss = torch.mean(discount[:, :-1] * critic_loss)

        actor_optimizer.zero_grad()
        actor_scaler.scale(actor_loss).backward()
        actor_scaler.unscale_(actor_optimizer)  # for clip grad

        policy_parameters = self.agent.efe_policy.parameters()
        if self.use_zclip_policy:
            grad_norm_actor = self.policy_zclip.step_with_parameters(policy_parameters)
        else:
            grad_norm_actor = (
                torch.nn.utils.clip_grad_norm_(
                    policy_parameters,
                    self.policy_grad_clip,
                )
                .detach()
                .cpu()
                .item()
            )
        actor_scaler.step(actor_optimizer)
        actor_scaler.update()
        actor_optimizer.zero_grad(set_to_none=True)
        actor_scheduler.step()

        critic_optimizer.zero_grad()
        critic_scaler.scale(critic_loss).backward()
        critic_scaler.unscale_(critic_optimizer)  # for clip grad

        critic_parameters = self.agent.efe_value.parameters()
        if self.use_zclip_policy:
            grad_norm_critic = self.critic_zclip.step_with_parameters(critic_parameters)
        else:
            grad_norm_critic = (
                torch.nn.utils.clip_grad_norm_(
                    critic_parameters,
                    self.policy_grad_clip,
                )
                .detach()
                .cpu()
                .item()
            )
        critic_scaler.step(critic_optimizer)
        critic_scaler.update()
        critic_optimizer.zero_grad(set_to_none=True)
        critic_scheduler.step()

        # Update slow critic
        with torch.no_grad():
            decay = self.update_slow_critic_decay
            for slow_param, param in zip(
                self.agent.efe_value_target.parameters(),
                self.agent.efe_value.parameters(),
            ):
                slow_param.data.copy_(
                    slow_param.data * decay + param.data * (1 - decay)
                )

        policy_dict["11_actor_loss"] = actor_loss.detach().cpu().item()
        policy_dict["12_actor_grad_norm"] = grad_norm_actor
        policy_dict["13_actor_norm_ratio"] = policy_norm_ratio
        policy_dict["21_critic_loss"] = critic_loss.detach().cpu().item()
        policy_dict["22_critic_grad_norm"] = grad_norm_critic

        return policy_dict

    def train_self_prior_transformer(
        self,
        self_prior_scaler: torch.amp.GradScaler,
        self_prior_optimizer,
        self_prior_scheduler,
        random_obs_visions,
        random_obs_proprios,
        obs_no_sticker,
        obs_sticker,
        get_sample=False,
    ):
        self.agent.train()
        self_prior_loss_dict = dict()

        freeze_list = [
            self.agent.obs_provider,
            self.agent.fe_world,
            self.agent.efe_policy,
            self.agent.efe_value,
            self.agent.efe_value_target,
        ]
        if self.conf.use_slow_self_prior:
            freeze_list.append(self.agent.self_prior_target)

        with FreezeParameters(freeze_list) and torch.autocast(
            device_type=self.conf.device,
            dtype=torch.bfloat16,
            enabled=self.conf.use_amp,
        ):

            def obs_to_x(vision, proprio):
                embed = self.agent.obs_provider(vision, proprio)
                _, post_sample = self.agent.fe_world.posterior_resample(embed)
                bos_x = self.agent.self_prior.prepare(post_sample)
                return bos_x

            bos_x = obs_to_x(random_obs_visions, random_obs_proprios)
            logits, targets = self.agent.self_prior.get_logits(bos_x)

            self_prior_loss = F.cross_entropy(
                rearrange(logits, "... C -> (...) C"),
                rearrange(targets, "... -> (...)"),
            )

            if self.conf.use_slow_self_prior:
                with torch.no_grad():
                    self.agent.self_prior_target.eval()
                    slow_logits, slow_targets = self.agent.self_prior_target.get_logits(
                        bos_x
                    )
                    log_p_teacher = slow_logits.log_softmax(dim=-1).detach()

                log_q_student = logits.log_softmax(dim=-1)
                slow_self_prior_loss = F.kl_div(
                    rearrange(log_q_student, "... C -> (...) C"),
                    rearrange(log_p_teacher, "... C -> (...) C"),
                    reduction="batchmean",
                    log_target=True,
                )
                self_prior_loss = self_prior_loss + slow_self_prior_loss

        self_prior_optimizer.zero_grad()
        self_prior_scaler.scale(self_prior_loss).backward()
        self_prior_scaler.unscale_(self_prior_optimizer)

        self_prior_parameters = self.agent.self_prior.parameters()
        if self.use_zclip_self_prior:
            grad_norm_self_prior = self.self_prior_zclip.step_with_parameters(
                self_prior_parameters
            )
        else:
            grad_norm_self_prior = torch.nn.utils.clip_grad_norm_(
                self_prior_parameters, self.conf.self_prior_grad_clip
            )
        self_prior_scaler.step(self_prior_optimizer)
        self_prior_scaler.update()
        self_prior_optimizer.zero_grad(set_to_none=True)
        self_prior_scheduler.step()

        if self.conf.use_slow_self_prior:
            with torch.no_grad():
                decay = self.conf.update_slow_self_prior_decay
                for slow_param, param in zip(
                    self.agent.self_prior_target.parameters(),
                    self.agent.self_prior.parameters(),
                ):
                    slow_param.data.copy_(
                        slow_param.data * decay + param.data * (1 - decay)
                    )

        if get_sample:
            assert obs_no_sticker is not None
            assert obs_sticker is not None

            with torch.no_grad():
                # Should be high.
                bos_x = obs_to_x(obs_no_sticker[0], obs_no_sticker[1])
                logits, targets = self.agent.self_prior.get_logits(bos_x)
                logprob_no_sticker = self.agent.self_prior.get_logprob(
                    logits, targets
                ).mean()

                # Should be low.
                bos_x = obs_to_x(obs_sticker[0], obs_sticker[1])
                logits, targets = self.agent.self_prior.get_logits(bos_x)
                logprob_sticker = self.agent.self_prior.get_logprob(
                    logits, targets
                ).mean()

                # Becomes positive.
                logprob_diff = logprob_no_sticker - logprob_sticker

                self_prior_loss_dict["32_logprob_diff"] = logprob_diff.item()
                self_prior_loss_dict["33_logprob_no_sticker"] = (
                    logprob_no_sticker.item()
                )
                self_prior_loss_dict["34_logprob_sticker"] = logprob_sticker.item()

                # Reconst
                n_samples = 10
                self_prior_samples = self.agent.self_prior.get_sample(n_samples)
                self_prior_vision, self_prior_proprio = (
                    self.agent.obs_provider.decode_from_feature(self_prior_samples)
                )
                self_prior_sample_dict = dict()
                self_prior_vision = (
                    torch.clamp(self_prior_vision + 0.5, 0, 1).cpu().numpy()
                )
                self_prior_sample_dict["01_self_prior_vision"] = self_prior_vision
        else:
            self_prior_sample_dict = dict()

        self_prior_loss_dict["31_self_prior_loss"] = self_prior_loss.item()
        self_prior_loss_dict["32_self_prior_grad_norm"] = grad_norm_self_prior

        return self_prior_loss_dict, self_prior_sample_dict
