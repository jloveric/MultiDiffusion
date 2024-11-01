from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from tqdm import tqdm

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torchvision.transforms as T
import argparse
import numpy as np
from PIL import Image
import click


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def get_views(panorama_height, panorama_width, window_size=64, stride=8):
    panorama_height /= 8
    panorama_width /= 8
    num_blocks_height = (panorama_height - window_size) // stride + 1
    num_blocks_width = (panorama_width - window_size) // stride + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views


class MultiDiffusion(nn.Module):
    def __init__(self, device, sd_version="2.0", hf_key=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f"[INFO] loading stable diffusion...")
        if hf_key is not None:
            print(f"[INFO] using hugging face custom model key: {hf_key}")
            model_key = hf_key
        elif self.sd_version == "2.1":
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == "2.0":
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == "1.5":
            model_key = "stablediffusionapi/architecturerealmix" #"runwayml/stable-diffusion-v1-5"
        else:
            model_key = (
                self.sd_version
            )  # For custom models or fine-tunes, allow people to use arbitrary versions
            # raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Create model
        self.vae = AutoencoderKL.from_pretrained(
            model_key, subfolder="vae", torch_dtype=torch.float16
        ).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_key, subfolder="tokenizer", torch_dtype=torch.float16
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_key, subfolder="text_encoder", torch_dtype=torch.float16
        ).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            model_key, subfolder="unet", torch_dtype=torch.float16
        ).to(self.device)

        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=torch.float16
        )

        print(f"[INFO] loaded stable diffusion!")

    @torch.no_grad()
    def get_random_background(self, n_samples):
        # sample random background with a constant rgb value
        backgrounds = torch.rand(n_samples, 3, device=self.device, dtype=torch.float16)[
            :, :, None, None
        ].repeat(1, 1, 512, 512)
        return torch.cat([self.encode_imgs(bg.unsqueeze(0)) for bg in backgrounds])

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        print('uncond_embeddings.shape', uncond_embeddings.shape, 'text_embeddings.shape', text_embeddings.shape)

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        print('text_embeddings.szi', text_embeddings.shape)
        return text_embeddings

    @torch.no_grad()
    def encode_imgs(self, imgs):
        print("imgs.dtype", imgs.dtype)
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def generate(
        self,
        masks,
        prompts,
        negative_prompts="",
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        bootstrapping=20,
        background_latent=None,
        start_time=0,
    ):

        # get bootstrapping backgrounds
        # can move this outside of the function to speed up generation. i.e., calculate in init
        bootstrapping_backgrounds = self.get_random_background(bootstrapping)

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(
            prompts, negative_prompts
        )  # [2 * len(prompts), 77, 768]

        # print("background latents", background_latent.dtype)

        # TODO: This may not work when the background image is giant, I'm not sure if the latent
        # will be the correct size.
        if background_latent is None:
            latent = torch.randn(
                (1, self.unet.config.in_channels, height // 8, width // 8),
                device=self.device,
                dtype=torch.float16,
            )
        else:
            latent = background_latent
        print('prompts', prompts)
        print("latent.shape", latent.shape)

        noise = latent.clone().repeat(len(prompts) - 1, 1, 1, 1)
        views = get_views(height, width)
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)

        self.scheduler.set_timesteps(num_inference_steps)
        start_iteration = int(start_time*num_inference_steps)

        with torch.autocast("cuda"):
            for i, t in enumerate(tqdm(self.scheduler.timesteps[start_iteration:])):
                count.zero_()
                value.zero_()

                for h_start, h_end, w_start, w_end in views:

                    # Narrow the masks and latents to the current block
                    masks_view = masks[:, :, h_start:h_end, w_start:w_end]
                    latent_view = latent[:, :, h_start:h_end, w_start:w_end].repeat(
                        len(prompts), 1, 1, 1
                    )

                    # print("masks_views", masks_view.shape, "latent_view", latent_view.shape)

                    
                    if i < bootstrapping:
                        bg = bootstrapping_backgrounds[
                            torch.randint(0, bootstrapping, (len(prompts) - 1,))
                        ]
                        #if background_latent is not None :
                        #    bg = background_latent

                        bg = self.scheduler.add_noise(
                            bg, noise[:, :, h_start:h_end, w_start:w_end], t
                        )
                        latent_view[1:] = latent_view[1:] * masks_view[1:] + bg * (
                            1 - masks_view[1:]
                        )
                    #print('masks[0]', torch.min(masks[0]),torch.max(masks[0]))
                    #print('masks[1]', torch.min(masks[1]),torch.max(masks[1]))

                    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    latent_model_input = torch.cat([latent_view] * 2)

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input, t, encoder_hidden_states=text_embeds
                    )["sample"]

                    # perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                    # compute the denoising step with the reference model
                    latents_view_denoised = self.scheduler.step(
                        noise_pred, t, latent_view
                    )["prev_sample"]

                    # Accumulate the latent values on this block
                    value[:, :, h_start:h_end, w_start:w_end] += (
                        latents_view_denoised * masks_view
                    ).sum(dim=0, keepdims=True)

                    # Count the number of times the pixel has been evaluated
                    count[:, :, h_start:h_end, w_start:w_end] += masks_view.sum(
                        dim=0, keepdims=True
                    )

                # Average the updated values of the entire latent space
                latent = torch.where(count > 0, value / count, value)

        # Img latents -> imgs
        imgs = self.decode_latents(latent)  # [1, 3, 512, 512]
        img = T.ToPILImage()(imgs[0].cpu())
        return img


def preprocess_mask(mask_path, h, w, device):
    mask = np.array(Image.open(mask_path).convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask).to(device)
    mask = torch.nn.functional.interpolate(mask, size=(h, w), mode="nearest")
    return mask


@click.command()
@click.option(
    "--mask_paths", type=str, multiple=True, required=True, help="List of mask paths"
)
@click.option("--bg_prompt", type=str, required=True, help="Background prompt")
@click.option(
    "--bg_negative",
    type=str,
    required=False,
    default="poor quality",
    help="Negative prompt for background",
)
@click.option(
    "--fg_prompts",
    type=str,
    multiple=True,
    required=True,
    help="List of foreground prompts",
)
@click.option(
    "--fg_negative",
    type=str,
    multiple=True,
    required=False,
    help="List of negative prompts for foreground",
)
@click.option(
    "--sd_version",
    type=click.Choice(["1.5", "2.0"]),
    default="2.0",
    help="Stable Diffusion version",
)
@click.option("--h", type=int, default=768, help="Height of the output image")
@click.option("--w", type=int, default=512, help="Width of the output image")
@click.option("--seed", type=int, default=0, help="Random seed")
@click.option("--steps", type=int, default=50, help="Number of diffusion steps")
@click.option(
    "--bootstrapping",
    type=int,
    default=20,
    help="Bootstrapping value for mask fidelity",
)
def main(
    mask_paths,
    bg_prompt,
    bg_negative,
    fg_prompts,
    fg_negative,
    sd_version,
    h,
    w,
    seed,
    steps,
    bootstrapping,
):
    print("Options:", locals())

    # Set the seed for reproducibility
    seed_everything(seed)

    # Initialize the device and model
    device = torch.device("cuda")
    sd = MultiDiffusion(device, sd_version)

    # Preprocess masks
    fg_masks = torch.cat(
        [preprocess_mask(mask_path, h // 8, w // 8, device) for mask_path in mask_paths]
    )
    bg_mask = 1 - torch.sum(fg_masks, dim=0, keepdim=True)
    bg_mask[bg_mask < 0] = 0
    masks = torch.cat([bg_mask, fg_masks])

    # Combine background and foreground prompts

    prompts = [bg_prompt] + list(fg_prompts)
    neg_prompts = [bg_negative] + list(fg_negative)

    # Generate and save the image
    img = sd.generate(
        masks, prompts, neg_prompts, h, w, steps, bootstrapping=bootstrapping
    )
    img.save("out.png")


if __name__ == "__main__":
    main()
