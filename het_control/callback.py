#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from typing import List

import torch
from tensordict import TensorDictBase, TensorDict

from benchmarl.experiment.callback import Callback
from het_control.models.het_control_mlp_empirical import HetControlMlpEmpirical
from het_control.snd import compute_behavioral_distance
from het_control.utils import overflowing_logits_norm
from het_control.rnd import compute_diversity_weights


def get_het_model(policy):
    model = policy.module[0]
    while not isinstance(model, HetControlMlpEmpirical):
        model = model[0]
    return model


class ADiCoCallback(Callback):
    """
    Callback that computes ADiCo diversity weights w(o) for the full
    collected batch, including RND predictor updates and learning progress
    estimation. Stores w(o) in the tensordict so it flows through to
    minibatch training.
    """

    def __init__(self, rnd_lr: float = 1e-3):
        super().__init__()
        self.rnd_lr = rnd_lr
        self.opt_dict = {}
        # EMA coefficient for running statistics (matches DiCo's tau default)
        self.tau_e = 0.01

    def on_setup(self):
        # Log ADiCo hyperparameters
        for group in self.experiment.group_map.keys():
            policy = self.experiment.group_policies[group]
            model = get_het_model(policy)
            if model.use_adico:
                self.experiment.logger.log_hparams(
                    adico_alpha=model.adico_alpha,
                    adico_beta=model.adico_beta,
                    rnd_lr=model.rnd_lr,
                )
                break

    def on_batch_collected(self, batch: TensorDictBase):
        for group in self.experiment.group_map.keys():
            policy = self.experiment.group_policies[group]
            model = get_het_model(policy)

            # Skip if ADiCo is not enabled or not applicable
            if not model.use_adico or model.desired_snd <= 0:
                continue

            # Get observations: shape [*batch_shape, n_agents, n_features]
            obs = batch.get((group, "observation"))
            original_shape = obs.shape[:-1]  # [*batch_shape, n_agents]
            flat_obs = obs.reshape(-1, obs.shape[-1])  # [N, n_features]

            # Ensure RND networks are on the correct device
            device = flat_obs.device
            if next(model.rnd.target.parameters()).device != device:
                model.rnd.to(device)

            # Create optimizer on first call
            if group not in self.opt_dict:
                self.opt_dict[group] = torch.optim.Adam(
                    model.rnd.predictor.parameters(), lr=model.rnd_lr
                )
            opt = self.opt_dict[group]

            target = model.rnd.target
            predictor = model.rnd.predictor

            # Step 1: Compute pre-update RND errors
            with torch.no_grad():
                target_features = target(flat_obs)  # [N, k]

            pred_features_before = predictor(flat_obs)  # [N, k]
            e_before = (
                (pred_features_before - target_features).pow(2).mean(dim=-1).detach()
            )  # [N]

            # Step 2: Update predictor with one gradient step
            loss = (pred_features_before - target_features.detach()).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Step 3: Compute post-update RND errors
            with torch.no_grad():
                pred_features_after = predictor(flat_obs)  # [N, k]
                e_after = (
                    (pred_features_after - target_features).pow(2).mean(dim=-1)
                )  # [N]

            # Step 4: Learning progress
            delta = e_before - e_after  # [N]

            # Step 5: Update running statistics with EMA
            batch_mean_e = e_before.mean()
            batch_std_e = e_before.std().clamp(min=1e-8)
            batch_std_delta = delta.std().clamp(min=1e-8)

            model.rnd_mean[:] = (
                (1 - self.tau_e) * model.rnd_mean + self.tau_e * batch_mean_e
            )
            model.rnd_std[:] = (
                (1 - self.tau_e) * model.rnd_std + self.tau_e * batch_std_e
            )
            model.rnd_delta_std[:] = (
                (1 - self.tau_e) * model.rnd_delta_std + self.tau_e * batch_std_delta
            )

            # Step 6: Compute diversity weights
            w = compute_diversity_weights(
                obs=flat_obs,
                target=target,
                predictor=predictor,
                rnd_mean=model.rnd_mean,
                rnd_std=model.rnd_std,
                alpha=model.adico_alpha,
                beta=model.adico_beta,
                delta=delta,
                delta_std=model.rnd_delta_std,
            )  # [N]

            # Reshape and store in tensordict
            w = w.reshape(*original_shape, 1)  # [*batch_shape, n_agents, 1]
            batch.set((group, "diversity_weight"), w)


class SndCallback(Callback):
    """
    Callback used to compute SND during evaluations
    """

    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        for group in self.experiment.group_map.keys():
            if not len(self.experiment.group_map[group]) > 1:
                # If agent group has 1 agent
                continue
            policy = self.experiment.group_policies[group]
            # Cat observations over time
            obs = torch.cat(
                [rollout.select((group, "observation")) for rollout in rollouts], dim=0
            )  # tensor of shape [*batch_size, n_agents, n_features]
            model = get_het_model(policy)

            # ADiCo: compute and store diversity weights for evaluation
            if model.use_adico and model.desired_snd > 0 and model.desired_snd != -1:
                raw_obs = obs.get((group, "observation"))
                w = model._compute_diversity_weight_online(raw_obs)
                obs.set((group, "diversity_weight"), w)

            agent_actions = []
            # Compute actions that each agent would take in this obs
            for i in range(model.n_agents):
                agent_actions.append(
                    model._forward(obs, agent_index=i, compute_estimate=False).get(
                        model.out_key
                    )
                )
            # Compute SND
            distance = compute_behavioral_distance(agent_actions, just_mean=True)
            self.experiment.logger.log(
                {f"eval/{group}/snd": distance.mean().item()},
                step=self.experiment.n_iters_performed,
            )


class NormLoggerCallback(Callback):
    """
    Callback to log some training metrics
    """

    def on_batch_collected(self, batch: TensorDictBase):
        for group in self.experiment.group_map.keys():
            keys_to_norm = [
                (group, "f"),
                (group, "g"),
                (group, "fdivg"),
                (group, "logits"),
                (group, "observation"),
                (group, "out_loc_norm"),
                (group, "estimated_snd"),
                (group, "scaling_ratio"),
                (group, "diversity_weight"),
            ]
            to_log = {}

            for key in keys_to_norm:
                value = batch.get(key, None)
                if value is not None:
                    to_log.update(
                        {"/".join(("collection",) + key): torch.mean(value).item()}
                    )
            self.experiment.logger.log(
                to_log,
                step=self.experiment.n_iters_performed,
            )


class TagCurriculum(Callback):
    """
    Tag curriculum used to freeze the green agents' policies during training
    """

    def __init__(self, simple_tag_freeze_policy_after_frames, simple_tag_freeze_policy):
        super().__init__()
        self.n_frames_train = simple_tag_freeze_policy_after_frames
        self.simple_tag_freeze_policy = simple_tag_freeze_policy
        self.activated = not simple_tag_freeze_policy

    def on_setup(self):
        # Log params
        self.experiment.logger.log_hparams(
            simple_tag_freeze_policy_after_frames=self.n_frames_train,
            simple_tag_freeze_policy=self.simple_tag_freeze_policy,
        )
        # Make agent group homogeneous
        policy = self.experiment.group_policies["agents"]
        model = get_het_model(policy)
        # Set the desired SND of the green agent team to 0
        # This is not important as the green agent team is composed of 1 agent
        model.desired_snd[:] = 0

    def on_batch_collected(self, batch: TensorDictBase):
        if (
            self.experiment.total_frames >= self.n_frames_train
            and not self.activated
            and self.simple_tag_freeze_policy
        ):
            del self.experiment.train_group_map["agents"]
            self.activated = True


class ActionSpaceLoss(Callback):
    """
    Loss to disincentivize actions outside of the space
    """

    def __init__(self, use_action_loss, action_loss_lr):
        super().__init__()
        self.opt_dict = {}
        self.use_action_loss = use_action_loss
        self.action_loss_lr = action_loss_lr

    def on_setup(self):
        # Log params
        self.experiment.logger.log_hparams(
            use_action_loss=self.use_action_loss, action_loss_lr=self.action_loss_lr
        )

    def on_train_step(self, batch: TensorDictBase, group: str) -> TensorDictBase:
        if not self.use_action_loss:
            return
        policy = self.experiment.group_policies[group]
        model = get_het_model(policy)
        if group not in self.opt_dict:
            self.opt_dict[group] = torch.optim.Adam(
                model.parameters(), lr=self.action_loss_lr
            )
        opt = self.opt_dict[group]
        loss = self.action_space_loss(group, model, batch)
        loss_td = TensorDict({"loss_action_space": loss}, [])

        loss.backward()

        grad_norm = self.experiment._grad_clip(opt)
        loss_td.set(
            f"grad_norm_action_space",
            torch.tensor(grad_norm, device=self.experiment.config.train_device),
        )

        opt.step()
        opt.zero_grad()

        return loss_td

    def action_space_loss(self, group, model, batch):
        # Select observation keys and diversity_weight if present
        keys_to_select = list(model.in_keys)
        dw_key = (model.agent_group, "diversity_weight")
        try:
            dw_val = batch.get(dw_key, None)
        except (KeyError, AttributeError):
            dw_val = None
        if dw_val is not None:
            keys_to_select.append(dw_key)

        logits = model._forward(
            batch.select(*keys_to_select), compute_estimate=True, update_estimate=False
        ).get(
            model.out_key
        )  # Compute logits from batch
        if model.probabilistic:
            logits, _ = torch.chunk(logits, 2, dim=-1)
        out_loc_norm = overflowing_logits_norm(
            logits, self.experiment.action_spec[group, "action"]
        )  # Compute how much they overflow outside the action space bounds

        # Penalise the maximum overflow over the agents
        max_overflowing_logits_norm = out_loc_norm.max(dim=-1)[0]

        loss = max_overflowing_logits_norm.pow(2).mean()
        return loss
