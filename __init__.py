"""Force PyTorch Attention - ComfyUI custom node.

Switches attention to PyTorch SDPA *only* for the model that passes through this node.
All other models keep the global attention backend (xformers / sage / ...).

Main use case: Qwen-Image 2.1 on RTX 50xx (Blackwell, sm_120) with xformers enabled.
xformers has no masked-attention kernel for sm_120 and fails with
"NotImplementedError: No operator found for memory_efficient_attention_forward".

Uses ComfyUI's built-in "optimized_attention_override" mechanism (transformer_options).
"""
import comfy.ldm.modules.attention as attn


class ForcePytorchAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches"
    DESCRIPTION = ("Use PyTorch attention for this model only (other models keep xformers/sage). "
                   "Fixes 'No operator found for memory_efficient_attention_forward' on RTX 50xx with xformers, e.g. Qwen-Image 2.1.")

    def patch(self, model):
        m = model.clone()
        to = m.model_options.setdefault("transformer_options", {})
        pytorch_attention = getattr(attn.attention_pytorch, "__wrapped__", attn.attention_pytorch)

        def override(func, *args, **kwargs):
            return pytorch_attention(*args, **kwargs)

        to["optimized_attention_override"] = override
        return (m,)


NODE_CLASS_MAPPINGS = {"ForcePytorchAttention": ForcePytorchAttention}
NODE_DISPLAY_NAME_MAPPINGS = {"ForcePytorchAttention": "Force PyTorch Attention (xformers fix)"}
