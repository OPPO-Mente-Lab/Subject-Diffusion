import os
import re
import torch
import argparse
from pytorch_lightning import (
    LightningModule,
    Trainer,
)
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
)
from model_utils import (
    add_module_args,
    configure_optimizers,
)
from diffusers import AutoencoderKL, DDPMScheduler, EulerDiscreteScheduler
from third_party.unet_2d_condition import UNet2DConditionModel
from torch.nn import functional as F
from tqdm.auto import tqdm
from typing import List, Optional, Union
from einops import rearrange
from third_party.localization_loss import unet_store_cross_attention_scores, BalancedL1Loss, get_object_localization_loss
from replace_clip_embedding import replace_clip_embeddings
from utils import save_model, numpy_to_pil
from torchvision.utils import save_image
import inspect
import cv2
from PIL import Image
from modules import MLP, GroundingNet
import open_clip
from transformers import CLIPTokenizer, CLIPTextModel
from torchvision import transforms
from torchvision.transforms.functional import crop
import gc
from custom_dataset import collate_fn, DataModuleCustom

IMAGE_256 = True
# 1 means definitely remove background, 0 means not remove background
RANDOM_REMOVE_BACKGROUND = 1
RANDOM_AUGMENTATION = True
RANDOM_BACKGROUND_COLOR = False
RESUME_ID = 0
RESUME_PATH = "/public_data/liangjunhao/GLIGEN/results_mul/stablediffusion_glyphdraw_token_multi_gligen_kv_bbox"
MAX_NUM_OBJECTS = 2
Object_localization = True

unet_config = {
    "act_fn": 'silu',
    "attention_head_dim": [
        5,
        10,
        20,
        20
    ],
    "block_out_channels": [
        320,
        640,
        1280,
        1280
    ],
    "center_input_sample": False,
    "cross_attention_dim": 1024,
    "down_block_types": [
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D"
    ],
    "downsample_padding": 1,
    "dual_cross_attention": False,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 5,
    "layers_per_block": 2,
    "mid_block_scale_factor": 1,
    "norm_eps": 1e-05,
    "norm_num_groups": 32,
    "num_class_embeds": None,
    "only_cross_attention": False,
    "out_channels": 4,
    "sample_size": 64,
    "up_block_types": [
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D"
    ],
    "use_linear_projection": True}


class StableDiffusion(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group(
            'OPPO Stable Diffusion Module')
        parser.add_argument('--tokenizer', default=None)

        parser.add_argument('--train_unet', default=True)
        parser.add_argument('--train_clip_visual', default=False)
        parser.add_argument('--train_clip_text', default=True)
        parser.add_argument('--train_transformer', default=True)
        parser.add_argument('--local_rank', default=-1,
                            type=int, help='node rank for distributed training')
        parser.add_argument('--strengthen', default=False)
        return parent_parser

    def __init__(self, args):
        super().__init__()

        self.tokenizer = CLIPTokenizer.from_pretrained(
            args.model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            os.path.join(args.model_path, "text_encoder"))

        self.vae = AutoencoderKL.from_pretrained(
            args.model_path, subfolder="vae")
        self.unet = UNet2DConditionModel(**unet_config)
        if RESUME_ID:
            self.unet.load_state_dict(torch.load(
                os.path.join(RESUME_PATH, f"unet_0_{RESUME_ID}/pytorch_model.bin")), strict=True)
        else:
            sd = torch.load(os.path.join(args.model_path,
                            "unet/diffusion_pytorch_model.bin"))
            if "state_dict" in list(sd.keys()):
                sd = sd["state_dict"]
            # keys = list(sd.keys())
            # Our model adds additional channels to the first layer to condition on an input image.
            # For the first layer, copy existing channel weights and initialize new channel weights to zero.

            input_key = "conv_in.weight"
            # channals add 1
            input_weight = self.unet.state_dict()[input_key]
            # torch.nn.init.normal_(input_weight)

            print(f"Manual init: {input_key}")
            input_weight.zero_()
            input_weight[:, :4, :, :].copy_(sd[input_key])
            sd[input_key] = input_weight
            missing_keys, unexpected_keys = self.unet.load_state_dict(
                sd, strict=False)
            print("Deleting key {} from state_dict.".format(input_key))
            assert unexpected_keys == []

        self.test_scheduler = EulerDiscreteScheduler.from_pretrained(
            args.model_path, subfolder="scheduler")
        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        self.save_hyperparameters(args)

        self.image_transforms_tensor = transforms.Compose(
            [
                transforms.Resize(
                    64, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomCrop(64),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    64, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomCrop(64),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.strengthen = args.strengthen

        self.model_clip, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained="/public_data/ma/models/ViT-H-14.pt")
        self.model_clip.visual.output_tokens = True
        self.mlp = MLP(1024, 1024, 1024, use_residual=False)
        # self.position_net = PositionNet(in_dim=1024, out_dim=1024)
        self.proj = GroundingNet(1280, 1024, 1024)
        if RESUME_ID:
            self.text_encoder.load_state_dict(torch.load(
                os.path.join(RESUME_PATH, f"clip_text_0_{RESUME_ID}/pytorch_model.bin"), map_location="cpu"))
            self.mlp.load_state_dict(torch.load(
                os.path.join(RESUME_PATH, f"mlp_0_{RESUME_ID}/pytorch_model.bin"), map_location="cpu"))
            self.proj.load_state_dict(torch.load(
                os.path.join(RESUME_PATH, f"proj_0_{RESUME_ID}/pytorch_model.bin"), map_location="cpu"))
        self.idx = 0
        # self.image_augmentation = transforms.Compose([transforms.RandomHorizontalFlip(
        #     p=0.5), transforms.RandomRotation(degrees=10), transforms.RandomPerspective(distortion_scale=0.5, p=0.5)])
        self.cross_attention_scores = {}
        self.unet = unet_store_cross_attention_scores(
            self.unet, self.cross_attention_scores, 5
        )
        if Object_localization:

            self.object_localization_loss_fn = BalancedL1Loss()
        self.image_infos = {"image_embedding": None, "image_token_mask": None}
        replace_clip_embeddings(self.text_encoder, self.image_infos)
        if self.hparams.train_unet:
            self.unet.train()
        else:
            self.unet.eval()
        if self.hparams.train_clip_visual:
            self.model_clip.train()
        else:
            self.model_clip.eval()
        if self.hparams.train_clip_text:
            self.text_encoder.train()
        else:
            self.text_encoder.eval()

    def _clear_cross_attention_scores(self):
        if hasattr(self, "cross_attention_scores"):
            keys = list(self.cross_attention_scores.keys())
            for k in keys:
                del self.cross_attention_scores[k]

        gc.collect()

    def setup(self, stage) -> None:
        if stage == 'fit':
            # 随便设置的，需要修改10^9/16/28=7,812,500
            self.total_steps = 2232142
            # self.total_steps = get_total_steps(self.trainer, self.hparams)
            print('Total steps: {}' .format(self.total_steps))

    def configure_optimizers(self):
        model_params = []
        if self.hparams.train_unet:
            params = []
            for name, p in self.unet.named_parameters():
                if ("transformer_blocks" in name) and ("attn2" in name) and (("to_k" in name) or ("to_v" in name)):
                    params.append(p)
                    p.requires_grad_(True)

                elif "fuser" in name:
                    params.append(p)
                    p.requires_grad_(True)
                    # params_u.append(p)
                    # names_u.append(name)
                elif "conv_in" in name:
                    params.append(p)
                    p.requires_grad_(True)
                    # names.append(name)
                else:
                    p.requires_grad_(False)

            model_params.append({'params':  iter(params)})

        if self.hparams.train_clip_visual:
            params = []
            for name, module in self.model_clip.named_parameters():
                if "visual" in name and ("30" in name or "31" in name):
                    params.append(module)
                    module.requires_grad_(True)
                elif name == "visual.proj" or "visual.ln_post" in name:
                    params.append(module)
                    module.requires_grad_(True)
                else:
                    module.requires_grad_(False)
            model_params.append({'params': iter(params)})
        if self.hparams.train_clip_text:
            model_params.append({'params': self.text_encoder.parameters()})
        model_params.append({'params': self.mlp.parameters()})
        # model_params.append({'params': self.position_net.parameters()})
        model_params.append({'params': self.proj.parameters()})
        return configure_optimizers(self, model_params=model_params)

    def encode_images(self, entity_images, image_token_idx_mask, bboxes, device):
        entity_images = rearrange(
            entity_images, "b n c h w -> (b n) c h w").to(device).half()
        image_embeddings_cls, image_embeddings = self.model_clip.half().encode_image(entity_images)
        image_embeddings_cls = self.mlp.half()(image_embeddings_cls)

        image_embeddings = rearrange(
            image_embeddings, "(b n) h d -> b n h d", n=MAX_NUM_OBJECTS)
        image_token_idx_mask = rearrange(
            image_token_idx_mask, "b n -> b n 1 1")
        image_embeddings = self.proj(
            image_embeddings, image_token_idx_mask, bboxes)

        return rearrange(image_embeddings_cls, "(b n) d -> b n d", n=MAX_NUM_OBJECTS), image_embeddings

    def encode_images_test(self, images, bboxes, device):
        image_tensor = []
        for image in images:
            image_tensor.append(self.preprocess(image).unsqueeze(0).to(device))
        image_embeddings_cls, image_embeddings = self.model_clip.to(
            device).half().encode_image(torch.cat(image_tensor).half())
        image_embeddings_cls = self.mlp.half()(image_embeddings_cls)
        image_token_idx_mask = torch.ones(
            1, 2, 1, 1, dtype=bool, device=device)
        return image_embeddings_cls, self.proj(image_embeddings.unsqueeze(0), image_token_idx_mask, bboxes.half())

    # 测试用
    def encode_images_mask(self, images, device):
        width_x = 512
        width_y = 512
        mask_img = torch.ones((1, width_x, width_y), device=device)
        bboxes = torch.zeros(1, 2, 4, device=device)
        transf = transforms.Compose([transforms.Resize(
            200, max_size=250, interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor()])
        img0 = transf(images[0])
        img1 = transf(images[1])
        mask_img[:, 100:100+img0.size(1), 10:10+img0.size(2)] = 0
        mask_img[:, 100:100+img1.size(1), 260:260+img1.size(2)] = 0
        bboxes[0, 0] = torch.tensor(
            (10, 100, 10+img0.size(2), 100+img0.size(1)), device=device)/512
        bboxes[0, 1] = torch.tensor(
            (260, 100, 260+img1.size(2), 100+img1.size(1)), device=device)/512
        mask_img_resize = transforms.Resize(
            (64, 64), interpolation=transforms.InterpolationMode.BICUBIC)(mask_img)
        # mask_tensor_resize = 1 - transforms.ToTensor()(mask_img_resize)
        return mask_img_resize, bboxes

    @torch.no_grad()
    def encode_prompt(self, prompt, entitys, images, bboxes, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        self.text_encoder = self.text_encoder.to(device)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        text_input_ids = text_inputs.input_ids

        # text_input_entity_ids = text_inputs_entitys.input_ids

        image_embeddings_cls, image_embeddings = self.encode_images_test(
            images, bboxes, device)
        self.image_infos["image_embedding"] = image_embeddings_cls
        self.image_infos["image_token_mask"] = (text_input_ids == 48136)
        encoder_hidden_states = self.text_encoder(text_input_ids)[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = encoder_hidden_states.shape
        encoder_hidden_states = encoder_hidden_states.repeat(
            1, num_images_per_prompt, 1)
        encoder_hidden_states = encoder_hidden_states.view(
            bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            self.image_infos["image_embedding"] = None
            self.image_infos["image_token_mask"] = None
            # text_embeddings = self.text_encoder(
            # text_input_ids.to(device), attention_mask=attention_mask)[0]
            uncond_embeddings = self.text_encoder(uncond_input.input_ids)
            uncond_embeddings = uncond_embeddings[0]

            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(
                1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(
                batch_size * num_images_per_prompt, seq_len, -1)

            encoder_hidden_states = torch.cat(
                [encoder_hidden_states, uncond_embeddings, uncond_embeddings])

        return encoder_hidden_states, image_embeddings

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(
            self.test_scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.test_scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def show_attention_map(self, attention_store, len_timesteps=50):
        resolution = 32
        out_map = []
        for key, value in attention_store.items():
            num_heads, size, num_tokens = value.size()
            value = value.view(num_heads, int(size**0.5), int(size**0.5),
                               num_tokens).mean(0).permute(2, 0, 1).unsqueeze(1)/len_timesteps

            if size != resolution**2:
                value = transforms.Resize((resolution, resolution))(value)
            out_map.append(value)
        return torch.cat(out_map, dim=1).mean(1, keepdim=True)

    @torch.no_grad()
    def log_imgs(
        self,
        device,
        inputs,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        num_images_per_prompt: Optional[int] = 1,
        **kwargs,
    ):
        prompts = inputs["prompt"]
        entitys = inputs["phrases"]
        batch_size = 1 if isinstance(prompts, str) else len(prompts)
        do_classifier_free_guidance = guidance_scale > 1.0

        image1 = Image.open(os.path.join(
            "/public_data/liangjunhao/GLIGEN/test", inputs["phrases"][0]+".jpg"))
        image2 = Image.open(os.path.join(
            "/public_data/liangjunhao/GLIGEN/test", inputs["phrases"][1]+".jpg"))
        latents_c2, bboxes = self.encode_images_mask([
            image1, image2], device)

        text_embeddings, image_embeddings = self.encode_prompt(prompts, entitys, [
            image1, image2], bboxes, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt)

        self.test_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.test_scheduler.timesteps

        shape = (batch_size * num_images_per_prompt,
                 4, height // 8, width // 8)

        latents = torch.randn(shape, generator=generator,
                              device=device, dtype=text_embeddings.dtype)
        latents = latents * self.test_scheduler.init_noise_sigma

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # latents_c1 = self.encode_images_vae([image], None, None, device)

        font_latents = latents_c2.to(device).half().unsqueeze(0)
        uncond_image_latents = torch.zeros_like(font_latents).to(device)
        font_latents = torch.cat(
            [font_latents, font_latents, uncond_image_latents], dim=0).to(device).half()
        attention_store = {}

        # gligen
        objects = image_embeddings
        uncond_objects = torch.zeros_like(objects)
        input_objects = torch.cat(
            [objects, objects, uncond_objects])  # 3B * 30 * 1024

        for i, t in enumerate(tqdm(timesteps)):

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat(
                [latents] * 3) if do_classifier_free_guidance else latents
            latent_model_input = self.test_scheduler.scale_model_input(
                latent_model_input, t)
            latent_model_input = torch.cat(
                [latent_model_input, font_latents], dim=1)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input.half(
            ), t, encoder_hidden_states=text_embeddings.half(), objs=input_objects.half()).sample
            for key, value in self.cross_attention_scores.items():
                if key in attention_store:
                    attention_store[key] += value
                else:
                    attention_store[key] = value

            if do_classifier_free_guidance:
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(
                    3)
                noise_pred = (
                    noise_pred_uncond
                    + guidance_scale * (noise_pred_text - noise_pred_image)
                    + image_guidance_scale *
                    (noise_pred_image - noise_pred_uncond)
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.test_scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs).prev_sample

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents.half()).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = numpy_to_pil(image)
        attention_map = self.show_attention_map(
            attention_store, len(timesteps))
        return image, attention_map


    def training_step(self, batch, batch_idx):

        # if self.hparams.train_transformer:
        #     # self.causal_transformer.train()
        #     self.fuse_module.train()
        # self.proj.train()
        # else:
        #     self.causal_transformer.requires_grad_(False)

        with torch.no_grad():
            latents = self.vae.encode(
                transforms.Normalize([0.5], [0.5])(batch["pixel_values"])).latent_dist.sample()
            latents = latents * 0.18215

        noise = torch.randn(latents.shape).to(latents.device)
        noise = noise.to(dtype=self.unet.dtype)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        noisy_latents = self.noise_scheduler.add_noise(
            latents, noise, timesteps)
        noisy_latents = noisy_latents.to(dtype=self.unet.dtype)
        # mask_font_pre = torch.stack(
        #     [torch.prod(t, dim=0).unsqueeze(0) for t in batch["masks"]])
        # mask_font_pre = torch.where(batch["object_segmaps"].sum(
        #     dim=1, keepdim=True) > 0, batch["pixel_values"], 1)
        # per image augmentation
        # for i in range(len(mask_font_pre)):
        #     mask_font_pre[i] = self.image_augmentation(mask_font_pre[i])
        mask_font_pre = torch.stack(
            [torch.prod(t, dim=0).unsqueeze(0) for t in batch["masks"]])
        # mask_font_pre = transforms.Grayscale()(mask_font_pre)
        mask_font = transforms.Resize(
            64, interpolation=transforms.functional._interpolation_modes_from_int(0))(mask_font_pre)
        image_embeddings_cls, image_embeddings = self.encode_images(
            batch["entity_images"], batch["image_token_idx_mask"], batch["bboxes"], latents.device)

        # replace embeddings index
        texts_input = self.tokenizer(batch['texts'],
                                     padding="max_length",
                                     max_length=self.tokenizer.model_max_length,
                                     truncation=True,
                                     return_tensors="pt",).to(
            latents.device)
        self.image_infos["image_embedding"] = image_embeddings_cls[batch["image_token_idx_mask"]]
        self.image_infos["image_token_mask"] = batch["image_token_mask"]

        text_embeddings = self.text_encoder(texts_input.input_ids)
        encoder_hidden_states = text_embeddings[0]
        # encoder_hidden_states_ori = encoder_hidden_states.clone()

        # gligen tokens
        # batch_new = []
        # mask = []
        # for box in batch["bboxes"]:
        #     if box.shape[0]==1:
        #         box = torch.cat([box,torch.zeros(box.shape).to(latents.device)])
        #         mask.append(torch.cat([torch.ones((1,)),torch.zeros((1,))]))
        #     else:
        #         mask.append(torch.ones((2,)))
        #     batch_new.append(box)
        # batch_news = torch.stack(batch_new)
        # masks = torch.stack(mask).to(latents.device)
        # objects = self.position_net.to(latents.device)(latents.device, batch_news, image_embeddings,masks) # B*N*1024
        objects = image_embeddings
        uncond_input = self.tokenizer(
            [""]*len(batch["texts"]),
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(
            latents.device)

        self.image_infos["image_embedding"] = None
        self.image_infos["image_token_mask"] = None
        uncond_embeddings = self.text_encoder(uncond_input.input_ids)
        null_prompt = uncond_embeddings[0]

        # latents_c1 = self.encode_images_vae(batch["pixel_values"], batch["pixel_segs"], batch["masks"], latents.device)
        latents_c2 = mask_font
        latents_c = latents_c2
        # To support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        uncond = 0.05
        uncond_image = 0.05
        random = torch.rand(latents.size(0), device=latents.device)
        prompt_mask = rearrange(random < 2 * uncond, "n -> n 1 1")
        input_mask = 1 - rearrange((random >= uncond_image).float()
                                   * (random < 3 * uncond_image).float(), "n -> n 1 1 1")
        # input_mask = 1 - rearrange((random >= uncond).float() * (random < 3 * uncond).float(), "n -> n 1 1 1")
        encoder_hidden_states = torch.where(
            prompt_mask, null_prompt, encoder_hidden_states)

        noise_pred = self.unet(torch.cat([noisy_latents.half(), input_mask.half(
        )*latents_c.half()], 1), timesteps, encoder_hidden_states, objs=objects.half()).sample
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        # loss_weights = (1 - mask_font) * 0.6 + torch.ones_like(mask_font)
        # loss = F.mse_loss(noise_pred, noise, reduction="none")*loss_weights
        loss = F.mse_loss(noise_pred, noise, reduction="none")
        loss = loss.mean([1, 2, 3]).mean()
        self.log("mse_loss", loss.item(),  on_epoch=False,
                 prog_bar=True, logger=True)
        if Object_localization:
            # b, max_num_objects, _, _
            batch["image_token_idx_mask"] = (
                ~prompt_mask.reshape(-1, 1))*batch["image_token_idx_mask"]
            object_segmaps = batch["object_segmaps"].repeat(1, 2, 1, 1)
            image_token_idx = torch.cat(
                (batch["image_token_idx"], batch["image_token_idx"]-2), dim=1)
            image_token_idx_mask = batch["image_token_idx_mask"].repeat(
                1, 2)
            image_token_idx[~image_token_idx_mask] = 0
            localization_loss = get_object_localization_loss(
                self.cross_attention_scores,
                object_segmaps,
                image_token_idx,
                image_token_idx_mask,
                self.object_localization_loss_fn,
            )
            loss = 0.01 * localization_loss + loss
            self.log("loc_loss", localization_loss.item(), on_epoch=False,
                     prog_bar=True, logger=True)
            self._clear_cross_attention_scores()

        self.log("lr", lr,  on_epoch=False, prog_bar=True, logger=True)

        if self.trainer.global_rank == 0:
            if (self.global_step+1) % 5000 == 0:
                print('saving model...')
                save_path = os.path.join(
                    args.default_root_dir, f'hf_out_{self.trainer.current_epoch}_{self.global_step}')
                save_path_unet = os.path.join(
                    args.default_root_dir, f'unet_{self.trainer.current_epoch}_{self.global_step}')
                if self.hparams.train_unet:
                    save_model(self.unet, save_path_unet)
                # save_model(self.causal_transformer, os.path.join(
                #     args.default_root_dir, f'transformer_{self.trainer.current_epoch}_{self.global_step}'))
                save_model(self.proj, os.path.join(args.default_root_dir,
                           f'proj_{self.trainer.current_epoch}_{self.global_step}'))
                save_model(self.mlp, os.path.join(args.default_root_dir,
                           f'mlp_{self.trainer.current_epoch}_{self.global_step}'))
                if self.hparams.train_clip_visual:
                    save_model(self.model_clip, os.path.join(
                        args.default_root_dir, f'clip_{self.trainer.current_epoch}_{self.global_step}'))
                if self.hparams.train_clip_text:
                    save_model(self.text_encoder, os.path.join(
                        args.default_root_dir, f'clip_text_{self.trainer.current_epoch}_{self.global_step}'))
                # # 生成测试图片
                # with torch.no_grad():
                #     inputs = {
                #         "prompt": "a dog and a cat playing on the beach, the dog is sks, the cat is sks",
                #         "phrases": ["dog_ori", "cat_ori"]
                #     }
                #     images, atten_map = self.log_imgs(
                #         latents.device, inputs, num_images_per_prompt=args.test_repeat)
                #     img_path = os.path.join(save_path, "test_images")
                #     save_images(images, img_path, [
                #                 inputs["prompt"]], args.test_repeat)
                #     save_image(atten_map, os.path.join(
                #         img_path, "attention_map.png"), normalize=True, pad_value=1)
                #     print("Test images saved to: {}".format(img_path))

        return {"loss": loss}

    def on_train_epoch_end(self):
        pass

    def on_load_checkpoint(self, checkpoint) -> None:
        # 兼容低版本lightning，低版本lightning从ckpt起来时steps数会被重置为0
        global_step_offset = checkpoint["global_step"]
        if 'global_samples' in checkpoint:
            self.consumed_samples = checkpoint['global_samples']
        self.trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser = add_module_args(args_parser)
    args_parser = DataModuleCustom.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = StableDiffusion.add_module_specific_args(args_parser)
    args = args_parser.parse_args()

    model = StableDiffusion(args)
    tokenizer = model.tokenizer

    datamoule = DataModuleCustom(
        args, tokenizer=tokenizer, collate_fn=collate_fn)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer.from_argparse_args(args,
                                         callbacks=[
                                             lr_monitor])
    device = torch.device(f"cuda:{trainer.local_rank}")
    model = model.to(device)
    trainer.fit(model, datamoule)
