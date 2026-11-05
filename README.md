# Force PyTorch Attention (xformers fix)

A tiny ComfyUI node that switches attention to **PyTorch SDPA for one model only**.
Every other model in your workflow keeps the global backend (xformers, sage, ...).

## Why
On **RTX 50xx (Blackwell)** with **xformers** installed, some models (e.g. **Qwen-Image 2.1**) crash with:

```
NotImplementedError: No operator found for `memory_efficient_attention_forward`
... requires device with capability <= (9, 0) but your GPU has capability (12, 0) (too new)
```

The usual fix is `--disable-xformers`, which changes attention for *all* models.
This node fixes only the model that needs it.

## Usage
`Load Diffusion Model` → **Force PyTorch Attention (xformers fix)** → sampler / guider.

No settings. Category: `model_patches`.

## How it works
Sets ComfyUI's built-in `transformer_options["optimized_attention_override"]` on a clone of the model,
so attention calls of that model go to `attention_pytorch`. Nothing global is changed.

## Install
ComfyUI Manager → search **Force PyTorch Attention**, or clone this repo into `ComfyUI/custom_nodes`.

## License
MIT
