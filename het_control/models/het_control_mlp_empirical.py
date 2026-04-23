#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Type, Sequence, Optional

import torch
import torch.nn.functional as F
from tensordict import TensorDictBase
from tensordict.nn import NormalParamExtractor
from torch import nn
from torchrl.modules import MultiAgentMLP

from benchmarl.models.common import Model, ModelConfig
from het_control.snd import compute_behavioral_distance
from het_control.utils import overflowing_logits_norm
from het_control.rnd import RNDContainer, compute_diversity_weights
from .utils import squash


class HetControlMlpEmpirical(Model):
    def __init__(
        self,
        activation_class: Type[nn.Module],
        num_cells: Sequence[int],
        desired_snd: float,
        probabilistic: bool,
        scale_mapping: Optional[str],
        tau: float,
        bootstrap_from_desired_snd: bool,
        process_shared: bool,
        use_adico: bool = False,
        adico_alpha: float = 1.0,
        adico_beta: float = 5.0,
        rnd_embed_dim: int = 64,
        rnd_hidden_dim: int = 64,
        rnd_lr: float = 1e-3,
        use_dndico: bool = False,
        dndico_alpha: float = 0.5,
        use_cadico: bool = False,
        cadico_alpha: float = 0.3,
        cadico_alpha_start: float = 0.5,
        cadico_alpha_end: float = 0.1,
        cadico_anneal_start_frac: float = 0.1,
        cadico_anneal_end_frac: float = 0.75,
        cadico_total_frames: int = 10_000_000,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_cells = num_cells
        self.activation_class = activation_class
        self.probabilistic = probabilistic
        self.scale_mapping = scale_mapping
        self.tau = tau
        self.bootstrap_from_desired_snd = bootstrap_from_desired_snd
        self.process_shared = process_shared

        self.use_adico = use_adico
        self.adico_alpha = adico_alpha
        self.adico_beta = adico_beta
        self.rnd_lr = rnd_lr

        self.use_dndico = use_dndico
        self.dndico_alpha = dndico_alpha

        self.use_cadico = use_cadico
        self.cadico_alpha = cadico_alpha
        self.cadico_alpha_start = cadico_alpha_start
        self.cadico_alpha_end = cadico_alpha_end
        self.cadico_anneal_start_frac = cadico_anneal_start_frac
        self.cadico_anneal_end_frac = cadico_anneal_end_frac
        self.cadico_total_frames = cadico_total_frames
        self.cadico_alpha_start = cadico_alpha_start
        self.cadico_alpha_end = cadico_alpha_end
        self.cadico_anneal_start_frac = cadico_anneal_start_frac
        self.cadico_anneal_end_frac = cadico_anneal_end_frac
        self.cadico_total_frames = cadico_total_frames

        self.register_buffer(
            name="desired_snd",
            tensor=torch.tensor([desired_snd], device=self.device, dtype=torch.float),
        )
        self.register_buffer(
            name="estimated_snd",
            tensor=torch.tensor([float("nan")], device=self.device, dtype=torch.float),
        )



        self.scale_extractor = (
            NormalParamExtractor(scale_mapping=scale_mapping)
            if scale_mapping is not None
            else None
        )

        self.input_features = self.input_leaf_spec.shape[-1]
        self.output_features = self.output_leaf_spec.shape[-1]

        self.shared_mlp = MultiAgentMLP(
            n_agent_inputs=self.input_features,
            n_agent_outputs=self.output_features,
            n_agents=self.n_agents,
            centralised=False,
            share_params=True,
            device=self.device,
            activation_class=self.activation_class,
            num_cells=self.num_cells,
        )

        agent_outputs = (
            self.output_features // 2 if self.probabilistic else self.output_features
        )
        self.agent_mlps = MultiAgentMLP(
            n_agent_inputs=self.input_features,
            n_agent_outputs=agent_outputs,
            n_agents=self.n_agents,
            centralised=False,
            share_params=False,
            device=self.device,
            activation_class=self.activation_class,
            num_cells=self.num_cells,
        )

        if self.use_adico:
            rnd_container = RNDContainer(
                input_dim=self.input_features,
                hidden_dim=rnd_hidden_dim,
                embed_dim=rnd_embed_dim,
                device=self.device,
            )
            object.__setattr__(self, "rnd", rnd_container)
            self.register_buffer("rnd_mean", torch.zeros(1, device=self.device, dtype=torch.float))
            self.register_buffer("rnd_std", torch.ones(1, device=self.device, dtype=torch.float))
            self.register_buffer("rnd_delta_std", torch.ones(1, device=self.device, dtype=torch.float))

    def _perform_checks(self):
        super()._perform_checks()
        if self.centralised or not self.input_has_agent_dim:
            raise ValueError(f"{self.__class__.__name__} can only be used for policies")
        if self.input_has_agent_dim and self.input_leaf_spec.shape[-2] != self.n_agents:
            raise ValueError("If the MLP input has the agent dimension, the second to last spec dimension should be the number of agents")
        if self.output_has_agent_dim and self.output_leaf_spec.shape[-2] != self.n_agents:
            raise ValueError("If the MLP output has the agent dimension, the second to last spec dimension should be the number of agents")

    def _forward(self, tensordict: TensorDictBase, agent_index: int = None,
                 update_estimate: bool = True, compute_estimate: bool = True) -> TensorDictBase:
        input = tensordict.get(self.in_key)
        shared_out = self.shared_mlp.forward(input)
        if agent_index is None:
            agent_out = self.agent_mlps.forward(input)
        else:
            agent_out = self.agent_mlps.agent_networks[agent_index].forward(input)

        shared_out = self.process_shared_out(shared_out)

        if (self.desired_snd > 0 and torch.is_grad_enabled()
                and compute_estimate and self.n_agents > 1):
            distance = self.estimate_snd(input)
            if update_estimate:
                self.estimated_snd[:] = distance.detach()
        else:
            distance = self.estimated_snd

        if self.desired_snd == 0:
            scaling_ratio = 0.0
        elif (self.desired_snd == -1 or distance.isnan().any() or self.n_agents == 1):
            scaling_ratio = 1.0
        else:
            scaling_ratio = torch.where(distance != self.desired_snd,
                                        self.desired_snd / distance, 1)

        # ADiCo: RND-based observation-dependent diversity weight
        if (self.use_adico and self.desired_snd > 0
                and self.desired_snd != -1 and not distance.isnan().any()):
            dw_key = (self.agent_group, "diversity_weight")
            try:
                w = tensordict.get(dw_key, None)
            except (KeyError, AttributeError):
                w = None
            if w is None and torch.is_grad_enabled():
                w = self._compute_diversity_weight_online(input)
            if w is not None:
                scaling_ratio = scaling_ratio * w.detach()

        # DN-DiCo: per-agent deviation-normalized diversity weighting
        if (self.use_dndico and self.desired_snd > 0
                and self.desired_snd != -1 and not distance.isnan().any()
                and agent_index is None and self.n_agents > 1):
            w_dn = self._compute_dndico_weight(agent_out)
            scaling_ratio = scaling_ratio * w_dn.detach()
            tensordict.set((self.agent_group, "diversity_weight"),
                           w_dn.detach().expand(*agent_out.shape[:-1], 1))

        # CADiCo: cosine-adaptive per-agent SND_des modulation
        if (self.use_cadico and self.desired_snd > 0
                and self.desired_snd != -1 and not distance.isnan().any()
                and agent_index is None and self.n_agents > 1):
            if self.probabilistic:
                shared_loc = shared_out.chunk(2, -1)[0]
            else:
                shared_loc = shared_out

            snd_local = self._compute_cadico_snd(agent_out, shared_loc, tensordict)
            scaling_ratio = torch.where(distance != snd_local,
                                        snd_local / distance,
                                        torch.ones_like(snd_local))
            tensordict.set((self.agent_group, "diversity_weight"), snd_local.detach())

        if self.probabilistic:
            shared_loc, shared_scale = shared_out.chunk(2, -1)
            agent_loc = shared_loc + agent_out * scaling_ratio
            out_loc_norm = overflowing_logits_norm(agent_loc, self.action_spec[self.agent_group, "action"])
            agent_scale = shared_scale
            out = torch.cat([agent_loc, agent_scale], dim=-1)
        else:
            out = shared_out + scaling_ratio * agent_out
            out_loc_norm = overflowing_logits_norm(out, self.action_spec[self.agent_group, "action"])

        tensordict.set((self.agent_group, "estimated_snd"),
                       self.estimated_snd.expand(tensordict.get_item_shape(self.agent_group)))
        tensordict.set((self.agent_group, "scaling_ratio"),
                       (torch.tensor(scaling_ratio, device=self.device).expand_as(out)
                        if not isinstance(scaling_ratio, torch.Tensor)
                        else scaling_ratio.expand_as(out)))
        tensordict.set((self.agent_group, "logits"), out)
        tensordict.set((self.agent_group, "out_loc_norm"), out_loc_norm)
        tensordict.set(self.out_key, out)

        return tensordict

    def _compute_diversity_weight_online(self, obs):
        original_shape = obs.shape[:-1]
        flat_obs = obs.reshape(-1, obs.shape[-1])
        device = obs.device
        if next(self.rnd.target.parameters()).device != device:
            self.rnd.to(device)
        w = compute_diversity_weights(
            obs=flat_obs, target=self.rnd.target, predictor=self.rnd.predictor,
            rnd_mean=self.rnd_mean, rnd_std=self.rnd_std,
            alpha=self.adico_alpha, beta=self.adico_beta, delta=None, delta_std=None)
        return w.reshape(*original_shape, 1)

    @torch.no_grad()
    def _compute_dndico_weight(self, agent_out):
        n_agents = agent_out.shape[-2]
        a_i = agent_out.unsqueeze(-2)
        a_j = agent_out.unsqueeze(-3)
        pairwise_dist = torch.linalg.norm(a_i - a_j, dim=-1)
        mask = ~torch.eye(n_agents, device=agent_out.device, dtype=torch.bool)
        d_i = (pairwise_dist * mask.float()).sum(dim=-1) / (n_agents - 1)
        d_mean = d_i.mean().clamp(min=1e-8)
        w_raw = d_i / d_mean
        w_blend = 1.0 + self.dndico_alpha * (w_raw - 1.0)
        w_clamped = w_blend.clamp(min=1.0 - self.dndico_alpha, max=1.0 + self.dndico_alpha)
        w = w_clamped / w_clamped.mean().clamp(min=1e-8)
        return w.unsqueeze(-1)

    def _get_cadico_alpha(self, tensordict=None):
        """Compute current alpha with linear annealing schedule.
        Reads the true frame count from tensordict (stamped by CADiCoAnnealCallback)."""
        frames = 0
        if tensordict is not None:
            try:
                tf = tensordict.get((self.agent_group, "total_frames"), None)
                if tf is not None:
                    frames = float(tf.flatten()[0].item())
            except (KeyError, AttributeError):
                pass
        if frames <= 0:
            return self.cadico_alpha
        frac = frames / max(self.cadico_total_frames, 1)
        if frac < self.cadico_anneal_start_frac:
            return self.cadico_alpha_start
        elif frac >= self.cadico_anneal_end_frac:
            return self.cadico_alpha_end
        else:
            t = (frac - self.cadico_anneal_start_frac) / (self.cadico_anneal_end_frac - self.cadico_anneal_start_frac)
            return self.cadico_alpha_start + t * (self.cadico_alpha_end - self.cadico_alpha_start)

    @torch.no_grad()
    def _compute_cadico_snd(self, agent_out, shared_loc, tensordict=None):
        """Per-agent, per-observation local SND_des via cosine disagreement.
        Uses annealed alpha: high early (exploration), low late (convergence).
        """
        alpha = self._get_cadico_alpha(tensordict)
        cos_sim = F.cosine_similarity(agent_out, shared_loc, dim=-1, eps=1e-8)
        disagreement = -cos_sim
        d_mean = disagreement.mean()
        d_std = disagreement.std().clamp(min=1e-8)
        d_norm = (disagreement - d_mean) / d_std
        snd_local = self.desired_snd + alpha * self.desired_snd * d_norm
        snd_max = min(2.0, float(self.desired_snd) * 2.0)
        snd_local = snd_local.clamp(min=0.0, max=snd_max)
        return snd_local.unsqueeze(-1)

    def process_shared_out(self, logits):
        if not self.probabilistic and self.process_shared:
            return squash(logits, action_spec=self.action_spec[self.agent_group, "action"], clamp=False)
        elif self.probabilistic:
            loc, scale = self.scale_extractor(logits)
            if self.process_shared:
                loc = squash(loc, action_spec=self.action_spec[self.agent_group, "action"], clamp=False)
            return torch.cat([loc, scale], dim=-1)
        else:
            return logits

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        if self.use_adico:
            sd["_rnd_target"] = self.rnd.target.state_dict()
            sd["_rnd_predictor"] = self.rnd.predictor.state_dict()
        return sd

    def load_state_dict(self, state_dict, *args, **kwargs):
        if self.use_adico:
            rnd_target_sd = state_dict.pop("_rnd_target", None)
            rnd_predictor_sd = state_dict.pop("_rnd_predictor", None)
            if rnd_target_sd is not None:
                self.rnd.target.load_state_dict(rnd_target_sd)
            if rnd_predictor_sd is not None:
                self.rnd.predictor.load_state_dict(rnd_predictor_sd)
        super().load_state_dict(state_dict, *args, **kwargs)

    def estimate_snd(self, obs):
        agent_actions = []
        for agent_net in self.agent_mlps.agent_networks:
            agent_actions.append(agent_net(obs))
        distance = (compute_behavioral_distance(agent_actions=agent_actions, just_mean=True)
                    .mean().unsqueeze(-1))
        if self.estimated_snd.isnan().any():
            distance = self.desired_snd if self.bootstrap_from_desired_snd else distance
        else:
            distance = (1 - self.tau) * self.estimated_snd + self.tau * distance
        return distance


@dataclass
class HetControlMlpEmpiricalConfig(ModelConfig):
    activation_class: Type[nn.Module] = MISSING
    num_cells: Sequence[int] = MISSING
    desired_snd: float = MISSING
    tau: float = MISSING
    bootstrap_from_desired_snd: bool = MISSING
    process_shared: bool = MISSING
    probabilistic: Optional[bool] = MISSING
    scale_mapping: Optional[str] = MISSING
    use_adico: bool = False
    adico_alpha: float = 1.0
    adico_beta: float = 5.0
    rnd_embed_dim: int = 64
    rnd_hidden_dim: int = 64
    rnd_lr: float = 1e-3
    use_dndico: bool = False
    dndico_alpha: float = 0.5
    use_cadico: bool = False
    cadico_alpha: float = 0.3
    cadico_alpha_start: float = 0.5
    cadico_alpha_end: float = 0.1
    cadico_anneal_start_frac: float = 0.1
    cadico_anneal_end_frac: float = 0.75
    cadico_total_frames: int = 10_000_000
    cadico_alpha_start: float = 0.5
    cadico_alpha_end: float = 0.1
    cadico_anneal_start_frac: float = 0.1
    cadico_anneal_end_frac: float = 0.75
    cadico_total_frames: int = 10_000_000

    @staticmethod
    def associated_class():
        return HetControlMlpEmpirical
