import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

import k_diffusion as K
import torch.nn as nn

from ldm.util                  import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.invoke.devices         import choose_torch_device

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(choose_torch_device())
    model.eval()
    return model

def export_as_gif(filename, images, frames_per_second=10, rubber_band=False):
    if rubber_band:
        images += images[2:-1][::-1]
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=1000 // frames_per_second,
        loop=0,
    )

def tensor_linspace(start, end, steps=10):
    """
    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out

def main():
    parser = argparse.ArgumentParser()

    prompt_1 = "A watercolor painting of a Golden Retriever at the beach"
    prompt_2 = "A still life DSLR photo of a bowl of fruit"

    parser.add_argument(
        "--prompts",
        type=str,
        nargs="?",
        default='{} | {}'.format(prompt_1, prompt_2),
        help="the prompts to interpolate between"
    )
    parser.add_argument(
        "--interpolation_steps",
        type=int,
        default=10,
        help="the number of steps to interpolate between prompts"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--klms",
        action='store_true',
        help="use klms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"


    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    seed_everything(opt.seed)

    device = torch.device(choose_torch_device())
    model  = model.to(device)

    #for klms
    model_wrap = K.external.CompVisDenoiser(model)
    class CFGDenoiser(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.inner_model = model

        def forward(self, x, sigma, uncond, cond, cond_scale):
            x_in = torch.cat([x] * 2)
            sigma_in = torch.cat([sigma] * 2)
            cond_in = torch.cat([uncond, cond])
            uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
            return uncond + (cond - uncond) * cond_scale

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompts = opt.prompts.split(" | ")
        assert prompts is not None
        data = [batch_size * [prompts[0]]]
        data2 = [batch_size * [prompts[1]]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            if (len(data) >= batch_size):
                data = list(chunk(data, batch_size))
            else:
                while (len(data) < batch_size):
                    data.append(data[-1])
                data = [data]

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        shape = [opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f]
        if device.type == 'mps':
            start_code = torch.randn(shape, device='cpu').to(device)
        else:
            torch.randn(shape, device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    if device.type in ['mps', 'cpu']:
        precision_scope = nullcontext # have to use f32 on mps
    with torch.no_grad():
        with precision_scope(device.type):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data2, desc="data2"):
                        print("data2 prompts", prompts)
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        prec2 = model.get_learned_conditioning(prompts)
                        
                    for prompts in tqdm(data, desc="data"):
                        print("data prompts", prompts)
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        prec1 = model.get_learned_conditioning(prompts)
                        c1 = torch.squeeze(prec1)
                        c2 = torch.squeeze(prec2)
                        # c1 = prec1
                        # c2 = prec2
                        # c = torch.linspace(c1, c2, opt.interpolation_steps)
                        c = tensor_linspace(c1, c2, opt.interpolation_steps)
                        print("c1", c1)
                        print("c2", c2)
                        print("c", c)
                        print("prec1 shape", prec1.shape)
                        print("prec2 shape", prec2.shape)
                        print("c1 shape", c1.shape)
                        print("c2 shape", c2.shape)
                        print("shape c", c.shape)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        batches = opt.interpolation_steps // opt.n_samples
                        batched_encodings = torch.split(c, batches)
                        final_images = []
                        print("shape", shape)
                        for batch in range(batches):
                          c_mixed = batched_encodings[batch][:, :, :, 0]
                          # c_mixed = torch.squeeze(c_mixed)
                          print("c_mixed shape", c_mixed.shape)
                          if not opt.klms:
                              samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                              conditioning=c_mixed,
                                                              batch_size=opt.n_samples,
                                                              shape=shape,
                                                              verbose=False,
                                                              unconditional_guidance_scale=opt.scale,
                                                              unconditional_conditioning=uc,
                                                              eta=opt.ddim_eta,
                                                              x_T=start_code)
                          else:
                              sigmas = model_wrap.get_sigmas(opt.ddim_steps)
                              if start_code:
                                  x = start_code
                              else:
                                  x = torch.randn([opt.n_samples, *shape], device=device) * sigmas[0] # for GPU draw
                              model_wrap_cfg = CFGDenoiser(model_wrap)
                              extra_args = {'cond': c_mixed, 'uncond': uc, 'cond_scale': opt.scale}
                              samples_ddim = K.sampling.sample_lms(model_wrap_cfg, x, sigmas, extra_args=extra_args)

                          x_samples_ddim = model.decode_first_stage(samples_ddim)
                          x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                          # final_images += [
                          #   Image.fromarray(255. * rearrange(img.cpu().numpy(), 'c h w -> h w c').astype(np.uint8))
                          #   for img in x_samples_ddim
                          # ]
                          
                          if not opt.skip_save:
                              print("saving")
                              for x_sample in x_samples_ddim:
                                  x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                  Image.fromarray(x_sample.astype(np.uint8)).save(
                                      os.path.join(sample_path, f"{base_count:05}.png"))
                                  base_count += 1

                          if not opt.skip_grid:
                            print("grid")
                            all_samples.append(x_samples_ddim)
                        export_as_gif("test.gif", final_images, rubber_band=True)
                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
