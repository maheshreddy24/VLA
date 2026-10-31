import torch
import torch.nn as nn

from .action_head import FlowMatchingActionHead
from .backbone import QwenVLBackbone


class VLA(nn.Module):
    """ GR00T-N1-style VLA: frozen Qwen3-VL (System 2) features from decoder layer `layer_id` are cross-attended by
    a flow-matching DiT action expert (System 1) that denoises a chunk of `horizon` actions. """

    def __init__(self, config):
        super().__init__()
        self.config = config
        m = config["model"]
        self.backbone = QwenVLBackbone(m["vlm_name"], m["layer_id"], freeze=m.get("freeze_vlm", True))
        h = m["action_head"]
        self.action_head = FlowMatchingActionHead(
            context_dim=self.backbone.hidden_size,
            action_dim=m["action_dim"],
            state_dim=m["state_dim"],
            horizon=m["horizon"],
            dim=h["dim"],
            depth=h["depth"],
            num_heads=h["num_heads"],
            mlp_ratio=h.get("mlp_ratio", 4.0),
            num_inference_steps=h.get("num_inference_steps", 10),
            timestep_sampling=h.get("timestep_sampling", "beta"),
        )

    def encode(self, vlm_inputs):
        context, mask = self.backbone(vlm_inputs)
        return context, mask

    def forward(self, batch):
        context, mask = self.encode(batch["vlm_inputs"])
        loss = self.action_head(context, mask, batch["state"], batch["actions"], batch.get("action_mask"))
        return {"loss": loss}

    @torch.no_grad()
    def predict_action(self, batch, num_steps=None):
        """ -> B, horizon, action_dim normalized actions """
        context, mask = self.encode(batch["vlm_inputs"])
        return self.action_head.sample(context, mask, batch["state"], num_steps)

    # Checkpoints only hold the trainable weights: the frozen VLM is reloaded from the hub.
    def trainable_state_dict(self):
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in names or not k.startswith("backbone.")}

    def load_trainable_state_dict(self, state_dict):
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        missing = [k for k in missing if not k.startswith("backbone.")]
        if missing or unexpected:
            raise RuntimeError(f"bad checkpoint: missing {missing[:5]}, unexpected {unexpected[:5]}")
