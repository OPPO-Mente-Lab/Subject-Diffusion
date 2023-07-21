import json
import os
import torch
import random


from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
from third_party.unet_2d_condition import UNet2DConditionModel as UNet2DConditionModel_GLIGEN
from PIL import Image
from tqdm.auto import tqdm
from typing import List, Optional, Union
from torchvision.utils import save_image
import inspect
from transformers import CLIPTokenizer, CLIPTextModel
import numpy as np
from utils import numpy_to_pil
from replace_clip_embedding import replace_clip_embeddings
import cv2
from einops import rearrange
import open_clip
from torchvision import transforms
from segment_anything import build_sam, SamPredictor
from third_party.localization_loss import unet_store_cross_attention_scores
from modules import MLP, GroundingNet
import argparse

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
MAX_NUMBER_OBJECTS = 2


class StableDiffusionTest():

    def __init__(self, model_path, unet_path, clip_text_path, mlp_path, proj_path, open_clip_path, sam_path, device):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            os.path.join(model_path, "text_encoder")).to(device)
        self.text_encoder.load_state_dict(
            torch.load(clip_text_path), strict=True)
        self.vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae").to(device)
        self.unet = UNet2DConditionModel_GLIGEN(**unet_config).to(device)
        self.unet.load_state_dict(torch.load(unet_path), strict=True)
        self.mlp = MLP(1024, 1024, 1024, use_residual=False)
        self.mlp.load_state_dict(torch.load(mlp_path), strict=True)
        self.proj = GroundingNet(1280, 1024, 1024, use_bbox=False)

        self.proj.load_state_dict(torch.load(proj_path, map_location="cpu"))
        self.unet_ori = UNet2DConditionModel.from_pretrained(
            model_path, subfolder="unet").to(device)
        self.test_scheduler = PNDMScheduler.from_pretrained(
            model_path, subfolder="scheduler")

        self.vae.eval()
        self.unet.eval()
        self.unet_ori.eval()

        self.model_clip, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained=open_clip_path)
        self.model_clip.visual.output_tokens = True
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    64, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomCrop(64),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5)
            ]
        )

        self.sam_model = build_sam(checkpoint=sam_path)
        self.sam_model.to(device=device)
        self.predictor = SamPredictor(self.sam_model)
        self.cross_attention_scores = {}
        self.unet = unet_store_cross_attention_scores(
            self.unet, self.cross_attention_scores, 5
        )
        self.image_infos = {}
        replace_clip_embeddings(self.text_encoder, self.image_infos)

    def encode_images_mask_old(self, device, images):
        masks = []
        bboxes = []
        for image in images:
            width_x = 512
            width_y = 512
            size = 350

            mask_img = torch.ones((1, 1, width_x, width_y))
            if isinstance(image, list):
                mask_img[:, :, 100:400, 50:200] = 0
                mask_img[:, :, 100:400, 300:500] = 0
                bbox = torch.tensor(
                    ([[50, 100, 200, 400], [300, 100, 500, 400]]))/512
                bboxes.append(bbox)
            else:
                seed1 = random.uniform(0.6, 0.8)
                seed2 = random.uniform(0.6, 0.8)
                chars_w, chars_h = int(size*seed1), int(size*seed2)
                chars_x = random.randint(10, 200)
                chars_y = random.randint(10, 200)
                mask_img[:, :, chars_y: chars_y + chars_h,
                         chars_x: chars_x + chars_w] = 0
                bbox = torch.tensor(
                    ([[chars_x, chars_y, chars_x + chars_w, chars_y + chars_h], [0, 0, 0, 0]]))/512
                bboxes.append(bbox)
            mask_img_resize = transforms.Resize(
                (64, 64), interpolation=transforms.InterpolationMode.NEAREST)(mask_img)
            masks.append(mask_img_resize)
        return torch.cat(masks).to(device), torch.stack(bboxes).to(device)

    def encode_images_test(self, images, bboxes, device):
        image_tensor = torch.empty(
            len(images), MAX_NUMBER_OBJECTS, 3, 224, 224, device=device)
        image_token_idx_mask = torch.zeros(
            len(images), MAX_NUMBER_OBJECTS, 1, 1, dtype=bool, device=device)
        for i in range(len(images)):
            if isinstance(images[i], list):
                image_token_idx_mask[i] = True
                for j in range(len(images[i])):
                    image_tensor[i][j] = self.preprocess(images[i][j])
            else:
                image_token_idx_mask[i, 0] = True
                image_tensor[i][0] = self.preprocess(images[i])
        image_embeddings_cls, image_embeddings = self.model_clip.to(
            device).encode_image(image_tensor.view(len(images)*MAX_NUMBER_OBJECTS, 3, 224, 224))
        image_embeddings_cls = self.mlp.to(device)(image_embeddings_cls)
        image_embeddings = rearrange(
            image_embeddings, "(b n) h d -> b n h d", n=MAX_NUMBER_OBJECTS)
        image_embeddings = self.proj.to(device)(
            image_embeddings, image_token_idx_mask, bboxes)
        return image_embeddings_cls.view(len(images), MAX_NUMBER_OBJECTS, -1), image_embeddings

    def generate_text_inputs(self, prompts, entities, device):
        image_token_mask = torch.zeros(
            (len(prompts), self.tokenizer.model_max_length), dtype=bool, device=device)
        image_token_idx_mask = torch.zeros(
            (len(prompts), MAX_NUMBER_OBJECTS), dtype=bool, device=device)
        batch_text_input = []
        for i, (prompt, entity) in enumerate(zip(prompts, entities)):
            if isinstance(entity, list):
                prompt = f"{prompt}, the {entity[0]} is sks, the {entity[1]} is sks"
                image_token_idx_mask[i] = True
            else:
                prompt = f"{prompt}, the {entity} is sks"
                image_token_idx_mask[i][0] = True
            text_input_ids = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids
            image_token_mask[i] = (text_input_ids == 48136)
            batch_text_input.append(prompt)
        return batch_text_input, image_token_mask, image_token_idx_mask

    @torch.no_grad()
    def encode_prompt(self, prompt, entities, images, bboxes, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, token=True):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        prompts, image_token_mask, image_token_idx_mask = self.generate_text_inputs(
            prompt, entities, device)
        # self.tokenizer.pad_token_id = 49407
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        # if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
        #     attention_mask = text_inputs.attention_mask.to(device)
        # else:
        #     attention_mask = None
        image_embeddings_cls, image_embeddings = self.encode_images_test(
            images, bboxes, device)

        self.image_infos["image_embedding"] = image_embeddings_cls[image_token_idx_mask]
        self.image_infos["image_token_mask"] = image_token_mask

        encoder_hidden_states = self.text_encoder(
            text_inputs.to(device).input_ids)[0]

        self.image_infos["image_embedding"] = None
        self.image_infos["image_token_mask"] = None
        ori_text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        # text_embeddings = self.text_encoder(
        #    **ori_text_input.to(device))[0]
        text_embeddings = self.text_encoder(
            ori_text_input.to(device).input_ids)[0]

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

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            self.image_infos["image_embedding"] = None
            self.image_infos["image_token_mask"] = None
            # uncond_embeddings = self.text_encoder(
            #    **uncond_input.to(device))[0]
            uncond_embeddings = self.text_encoder(
                uncond_input.to(device).input_ids)[0]
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(
                1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(
                batch_size * num_images_per_prompt, seq_len, -1)

            text_embeddings = torch.cat(
                [uncond_embeddings, text_embeddings])
            encoder_hidden_states = torch.cat(
                [encoder_hidden_states, uncond_embeddings, uncond_embeddings])
        return encoder_hidden_states, text_embeddings, image_embeddings, image_embeddings_cls

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

    def find_main_entity(self, masks):
        center = masks.shape[1]/2, masks.shape[2]/2
        threshold = masks.shape[1]*masks.shape[2]*0.01
        dists = np.empty(len(masks), dtype=np.float32)
        for idx, mask in enumerate(masks):
            coords = np.where(mask)
            if len(coords[0]) < threshold:
                dists[idx] = np.Inf
            else:
                dist = ((coords[0]-center[0])**2).mean() + \
                    ((coords[1]-center[1])**2).mean()
                dists[idx] = dist
        min_channel = dists.argmin()
        return min_channel

    def fill_cavity_2(self, input_mask):
        # slow but accurate
        ret_mask = input_mask.astype("uint8")*255
        contour, hier = cv2.findContours(
            ret_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            cv2.drawContours(ret_mask, [cnt], 0, 255, -1)
        return ret_mask.astype("bool")

    @torch.no_grad()
    def log_imgs(
        self,
        device,
        inputs,
        dataset_path,
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
        prompts = []
        images = []
        entitys = []
        phrases = []
        for t in inputs:
            prompts.append(t["prompt"])
            entitys.append(t["entities"])
            phrases.append(t["phrases"])
            if isinstance(t["phrases"], list):
                tmp_images = []
                for phrase in t["phrases"]:

                    subject_path = os.path.join(
                       dataset_path, phrase)
                    tmp_images.append(Image.open(os.path.join(
                        subject_path, sorted(os.listdir(subject_path))[0])))
                images.append(tmp_images)
            else:
                subject_path = os.path.join(
                    dataset_path, t["phrases"])
                images.append(Image.open(os.path.join(
                    subject_path, sorted(os.listdir(subject_path))[0])))
        # latents_c2 = torch.zeros(len(images),1,64,64,device=device)
        # sam

        def sam(image, entity):

            numpy_image = np.array(image)
            self.predictor.set_image(numpy_image)
            # transformed_boxes = self.predictor.transform.apply_boxes(np.asarray(bbox).reshape(1,-1), image.size)

            # masks, iou_predictions, low_res_masks = self.predictor.predict(point_coords = None,point_labels = None,box = transformed_boxes,multimask_output = False,)
            coord = np.array(image.size).reshape(1, 2)//2
            masks, iou_predictions, low_res_masks = self.predictor.predict(
                point_coords=coord, point_labels=np.array([1]))
            # masks,iou_predictions, low_res_masks = self.predictor.predict()
            channel = self.find_main_entity(masks)
            mask = masks[channel]
            mask = self.fill_cavity_2(mask)
            # channel = iou_predictions.argmax()
            # channel = masks.sum((-1,-2)).argmax()
            # set 0
            # numpy_image = mask[:, :, np.newaxis].repeat(3, axis=2)*numpy_image
            # set 1
            numpy_image = np.where(
                mask[:, :, np.newaxis].repeat(3, axis=2), numpy_image, 255)
            y, x = np.where(mask)
            ret_image = Image.fromarray(
                numpy_image[y.min():y.max(), x.min():x.max(), :])
            return ret_image
        for idx in range(len(images)):

            if isinstance(images[idx], list):
                for i in range(len(images[idx])):
                    images[idx][i] = sam(images[idx][i], phrases[idx][i])
            else:
                if phrases[idx] == 'coat' or phrases[idx] == 'overcoat':
                    continue
                images[idx] = sam(images[idx], phrases[idx])

            # Image.fromarray(masks.transpose(1,2,0).astype(np.uint8)).save(f"masks/{entitys[idx]}-00_mask.png")
        batch_size = 1 if isinstance(prompts, str) else len(prompts)
        do_classifier_free_guidance = guidance_scale > 0.5

        latents_c2, bboxes = self.encode_images_mask_old(device, images)

        text_embeddings, text_embeddings_ori, image_embeddings, image_embeddings_cls = self.encode_prompt(
            prompts, entitys, images, bboxes, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt)
        # text_embeddings_ori = self.encode_prompt(
        #     prompts, entitys, images, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, token=False)

        self.test_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.test_scheduler.timesteps

        shape = (batch_size * num_images_per_prompt,
                 4, height // 8, width // 8)

        latents = torch.randn(shape, generator=generator,
                              device=device, dtype=text_embeddings.dtype)
        latents = latents * self.test_scheduler.init_noise_sigma

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        font_latents = latents_c2
        uncond_image_latents = torch.zeros_like(font_latents).to(device)
        font_latents = torch.cat(
            [font_latents, font_latents, uncond_image_latents], dim=0).to(device).half()
        self_timesteps = 50
        attention_store = {}

        objects = image_embeddings
        uncond_objects = torch.zeros_like(objects)
        input_objects = torch.cat([objects, objects, uncond_objects])
        for i, t in enumerate(tqdm(timesteps)):

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat(
                [latents] * 3) if do_classifier_free_guidance else latents
            latent_model_input = self.test_scheduler.scale_model_input(
                latent_model_input, t)
            latent_model_input = torch.cat(
                [latent_model_input, font_latents], dim=1)
            if i < self_timesteps:
                noise_pred = self.unet.half()(latent_model_input.half(), t,
                                              encoder_hidden_states=text_embeddings.half(), objs=input_objects.half()).sample
                if do_classifier_free_guidance:
                    noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(
                        3)
                    noise_pred = (
                        noise_pred_uncond
                        + guidance_scale * (noise_pred_text - noise_pred_image)
                        + image_guidance_scale *
                        (noise_pred_image - noise_pred_uncond)
                    )
                for key, value in self.cross_attention_scores.items():
                    if key in attention_store:
                        attention_store[key] += value
                    else:
                        attention_store[key] = value

            else:
                latent_model_input = torch.cat(
                    [latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.test_scheduler.scale_model_input(
                    latent_model_input, t)
                noise_pred = self.unet_ori.half()(latent_model_input.half(
                ), t, encoder_hidden_states=text_embeddings_ori.half()).sample
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * \
                        (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.test_scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs).prev_sample

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = numpy_to_pil(image)
        resolution = 32
        out_map = []
        for key, value in attention_store.items():
            num_heads, size, num_tokens = value.size()
            value = value.view(num_heads, int(size**0.5), int(size**0.5),
                               num_tokens).mean(0).permute(2, 0, 1).unsqueeze(1)/self_timesteps

            if size != resolution**2:
                value = transforms.Resize((resolution, resolution))(value)
            out_map.append(value)
        return image, torch.cat(out_map, dim=1).mean(1, keepdim=True)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        '--prompts_path', default="./test_prompts/test_glyphdraw_multi2.json", type=str)
    args_parser.add_argument('--model_idx', default="449999", type=int)
    args_parser.add_argument('--model_path', default="./checkpoints", type=str)
    args_parser.add_argument(
        '--base_model_path', default="./model_base", type=str)
    args_parser.add_argument(
        '--sam_path', default="./sam_vit_h_4b8939.pth", type=str)
    args_parser.add_argument(
        '--open_clip_path', default="./ViT-H-14.pt", type=str)
    args_parser.add_argument('--output_path', default="./output", type=str)
    args_parser.add_argument('--guidance_scale', default=7.5, type=float)
    args_parser.add_argument('--image_guidance_scale', default=1.5, type=float)
    args_parser.add_argument('--batch_size', default=1, type=int)
    args_parser.add_argument('--dataset_path', default="./dreambooth", type=str)

    args = args_parser.parse_args()

    with open(args.prompts_path, "r", encoding='utf-8') as f:
        inputs = json.load(f)

    unet_path = os.path.join(
        args.model_path, f"unet_0_{args.model_idx}/pytorch_model.bin")
    clip_text_path = os.path.join(
        args.model_path, f"clip_text_0_{args.model_idx}/pytorch_model.bin")
    mlp_path = os.path.join(
        args.model_path, f"mlp_0_{args.model_idx}/pytorch_model.bin")
    proj_path = os.path.join(
        args.model_path, f"proj_0_{args.model_idx}/pytorch_model.bin")

    print("out_path:", args.output_path)
    os.makedirs(args.output_path, exist_ok=True)

    device = torch.device("cuda")
    lgp_test = StableDiffusionTest(
        args.base_model_path, unet_path, clip_text_path, mlp_path, proj_path, args.open_clip_path, args.sam_path, device)

    batch = args.batch_size
    inputs = inputs*batch
    raw_name = [t["prompt"] for t in inputs]
    for i in range(0, len(raw_name), batch):
        text_batch = inputs[i:(i + batch)]
        images, attention_map = lgp_test.log_imgs(
            device, text_batch,args.dataset_path, guidance_scale=args.guidance_scale, image_guidance_scale=args.image_guidance_scale)
        for j in range(batch):
            if j > len(images)-1:
                continue
            images[j].save(os.path.join(args.output_path, "{}-00_{}.jpg".format(text_batch[j]["phrases"],
                                                                                raw_name[i + j].replace(" ", "-")+str(i+j))), normalize=True)
