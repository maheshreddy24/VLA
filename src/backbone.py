import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class QwenVLBackbone(nn.Module):
    """ Frozen Qwen3-VL truncated after decoder layer `layer_id`.

    Returns the same features as `hidden_states[layer_id]` of the full model (index 0 = token embeddings), but
    only runs the vision tower plus the first `layer_id` decoder layers: the later layers, the final norm and
    the lm head are dropped. """

    def __init__(self, model_name, layer_id, dtype=torch.bfloat16, freeze=True):
        super().__init__()
        vlm = Qwen3VLForConditionalGeneration.from_pretrained(model_name, dtype=dtype)
        self.model = vlm.model  # Qwen3VLModel: vision tower + language model, no lm_head
        lm = self.model.language_model
        assert 0 < layer_id <= len(lm.layers), f"layer_id {layer_id} not in [1, {len(lm.layers)}]"
        lm.layers = lm.layers[:layer_id]
        lm.norm = nn.Identity()  # hidden_states[layer_id] is the raw residual stream, before the final norm
        self.hidden_size = lm.config.hidden_size
        self.layer_id = layer_id

        self.freeze = freeze
        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def forward(self, vlm_inputs):
        """ vlm_inputs: processor output (input_ids, attention_mask, pixel_values, image_grid_thw, ...)
        -> features B, M, hidden_size and mask B, M (True = real token) """
        with torch.set_grad_enabled(not self.freeze and torch.is_grad_enabled()):
            out = self.model(**vlm_inputs, use_cache=False)
        return out.last_hidden_state, vlm_inputs["attention_mask"].bool()


def load_processor(model_name):
    return AutoProcessor.from_pretrained(model_name)
