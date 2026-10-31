import torch
import torch.nn as nn

from .DiT import DDTBlock, FinalLayer, GaussianFourierEmbedding, RMSNorm, SwiGLUFFN


class FlowMatchingActionHead(nn.Module):
    """ GR00T-N1-style action expert: a DiT over [noisy action chunk tokens, state token] that cross-attends to
    VLM features and is trained with rectified flow.

    Convention (as in the original prototype): x_t = (1 - t) * noise + t * action, so t = 0 is pure noise and
    t = 1 is data. The network predicts the velocity v = action - noise; sampling integrates t from 0 to 1. """

    def __init__(self, context_dim, action_dim, state_dim, horizon, dim=512, depth=12, num_heads=8, mlp_ratio=4.0,
                 num_inference_steps=10, timestep_sampling="beta"):
        super().__init__()
        self.action_dim, self.state_dim, self.horizon = action_dim, state_dim, horizon
        self.num_inference_steps = num_inference_steps
        self.timestep_sampling = timestep_sampling

        self.context_proj = nn.Sequential(RMSNorm(context_dim), nn.Linear(context_dim, dim))
        self.action_in = nn.Linear(action_dim, dim)
        self.state_in = nn.Sequential(nn.Linear(state_dim, dim), SwiGLUFFN(dim, int(2 / 3 * dim * mlp_ratio)))
        self.time_embed = GaussianFourierEmbedding(dim)
        self.blocks = nn.ModuleList([DDTBlock(dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.final = FinalLayer(dim, action_dim)
        self.initialize_weights()

    def initialize_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_init)
        # AdaLN-Zero: every block starts as the identity, the output layer starts at zero velocity.
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    def sample_time(self, batch_size, device):
        if self.timestep_sampling == "beta":
            # pi0 / GR00T: emphasize the noisy end (t near 0 here), never exactly t = 1.
            beta = torch.distributions.Beta(torch.tensor(1.5, device=device), torch.tensor(1.0, device=device))
            return 0.999 * (1 - beta.sample((batch_size,)))
        return torch.rand(batch_size, device=device)

    def predict_velocity(self, x_t, t, state, context, context_mask):
        """ x_t: B, H, A; t: B; state: B, S; context: B, M, D_ctx (already projected); -> B, H, A """
        x = torch.cat([self.action_in(x_t), self.state_in(state).unsqueeze(1)], dim=1)  # state token last
        c = self.time_embed(t)  # B, dim
        for block in self.blocks:
            x = block(x, c, context, context_mask)
        return self.final(x[:, :self.horizon], c)

    def forward(self, context, context_mask, state, actions, action_mask=None):
        """ Flow-matching loss. actions: B, H, A (normalized); action_mask: B, H (False on padded steps). """
        context = self.context_proj(context)
        t = self.sample_time(actions.shape[0], actions.device)
        noise = torch.randn_like(actions)
        t_ = t[:, None, None]
        x_t = (1 - t_) * noise + t_ * actions
        v_pred = self.predict_velocity(x_t, t, state, context, context_mask)
        err = ((v_pred.float() - (actions - noise).float()) ** 2).mean(-1)  # B, H
        if action_mask is None:
            return err.mean()
        action_mask = action_mask.float()
        return (err * action_mask).sum() / action_mask.sum().clamp(min=1)

    @torch.no_grad()
    def sample(self, context, context_mask, state, num_steps=None):
        """ Euler integration from noise (t = 0) to actions (t = 1). -> B, H, A normalized actions """
        num_steps = num_steps or self.num_inference_steps
        context = self.context_proj(context)
        B = state.shape[0]
        x = torch.randn(B, self.horizon, self.action_dim, device=state.device, dtype=context.dtype)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=state.device, dtype=context.dtype)
            x = x + dt * self.predict_velocity(x, t, state, context, context_mask).to(x.dtype)
        return x
