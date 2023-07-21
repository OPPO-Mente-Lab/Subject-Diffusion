import json
import os
import torch
import random
import argparse

from third_party.diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel as UNet2DConditionModel_GLIGEN
from diffusers import UNet2DConditionModel
from torch.nn import functional as F
from PIL import Image, ImageOps
from tqdm.auto import tqdm
from typing import Callable, List, Optional, Union
from torchvision.utils import save_image
import inspect
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel
import numpy as np
from replace_clip_embedding import replace_clip_embeddings
import cv2
from einops import rearrange
import open_clip
from torchvision import transforms
from segment_anything import build_sam, SamPredictor
# from localization_loss import unet_gligen_store_cross_attention_scores
from modules import MLP, GroundingNet
###############
import base64
from io import BytesIO
import traceback
###############

# help functions
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def base64_to_cv2_img(base64_str, image_path=None):
    byte_data = base64.b64decode(base64_str)
    nparr = np.fromstring(byte_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def base64_to_pil(base64_str, image_path=None):
    try:
        byte_data = base64.b64decode(base64_str)
        image_data = BytesIO(byte_data)
        # img = Image.open(image_data)
        # if image_path:
        #     img.save(image_path)
    except Exception:
        traceback.print_exc()
        print("Convert base64 to pil failed. Return None.")
        return None
    return image_data


def pil_to_base64(imgs):
    # img = Image.open(image_path)
    results = []
    for img in imgs:
        base64_str = ""
        try:
            output_buffer = BytesIO()
            img.save(output_buffer, format='JPEG')
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data)
        except Exception:
            traceback.print_exc()
            print("Convert pil to base64 failed. Use empty string.")
        finally:
            results.append(base64_str)
    return results


def pad_image(image):
    pad_size = max(image.size)
    image = ImageOps.pad(image, (pad_size, pad_size), color="white")
    return image


class Text2Img:
    def __init__(self, device):
        # ids = 179999
        # method_name = "glyphdraw_multi_2"
        # self.patch_512 = True
        # self.use_bbox = True

        # # ids = 215999 #29999
        # # method_name = "glyphdraw_multi"

        # # ids = 419999  # 29999
        # # method_name = "stablediffusion_glyphdraw_token_multi_gligen"
        # # self.patch_512 = False
        
        # self.unet_path = f"/public_data/ma/code/GLIGEN-master/results_mul/{method_name}/unet_0_{ids}/pytorch_model.bin"
        # self.clip_text_path = f"/public_data/ma/code/GLIGEN-master/results_mul/{method_name}/clip_text_0_{ids}/pytorch_model.bin"
        # self.mlp_path = f"/public_data/ma/code/GLIGEN-master/results_mul/{method_name}/mlp_0_{ids}/pytorch_model.bin"
        # self.proj_path = f"/public_data/ma/code/GLIGEN-master/results_mul/{method_name}/proj_0_{ids}/pytorch_model.bin"


        ids = 449999  #159999  ##464999  ##464999*(12*8) + 159999*(12*8*2)
        method_name = "stablediffusion_glyphdraw_token_multi_gligen_kv_bbox"
        self.patch_512 = True
        self.use_bbox = True


        self.unet_path = f"/public_data/liangjunhao/GLIGEN/results_mul/{method_name}/unet_0_{ids}/pytorch_model.bin"
        self.clip_text_path = f"/public_data/liangjunhao/GLIGEN/results_mul/{method_name}/clip_text_0_{ids}/pytorch_model.bin"
        self.mlp_path = f"/public_data/liangjunhao/GLIGEN/results_mul/{method_name}/mlp_0_{ids}/pytorch_model.bin"
        self.proj_path = f"/public_data/liangjunhao/GLIGEN/results_mul/{method_name}/proj_0_{ids}/pytorch_model.bin"

        self.sam_checkpoint = "/public_data/ma/models/sam/sam_vit_h_4b8939.pth"
        self.model_path = "/public_data/ma/stable_models/model_base"

        self.device = device
        self.TOKEN_SCALE = 0.3
        self.inner_coords= True
        self.MAX_NUMBER_OBJECTS = 2
        self.unet_config = {
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

    def load(self):

        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_path, subfolder="tokenizer")
        self.text_encoder_ori = CLIPTextModel.from_pretrained(os.path.join(self.model_path, "text_encoder")).to(self.device)
        self.text_encoder = CLIPTextModel.from_pretrained(os.path.join(self.model_path, "text_encoder")).to(self.device)
        self.text_encoder.load_state_dict(torch.load(self.clip_text_path), strict=True)
        self.vae = AutoencoderKL.from_pretrained(self.model_path, subfolder="vae").to(self.device)
        self.unet = UNet2DConditionModel_GLIGEN(**self.unet_config).to(self.device)
        self.unet.load_state_dict(torch.load(self.unet_path), strict=True)
        self.mlp = MLP(1024, 1024, 1024, use_residual=False)
        self.mlp.load_state_dict(torch.load(self.mlp_path), strict=True)
        if self.patch_512:
            self.proj = GroundingNet(1280, 1024, 1024,use_bbox=self.use_bbox)
        else:
            self.proj = MLP(2560, 1024, 1280, use_residual=False)
        self.proj.load_state_dict(torch.load(
            self.proj_path, map_location="cpu"))
        self.unet_ori = UNet2DConditionModel.from_pretrained(
            self.model_path, subfolder="unet").to(self.device)
        self.test_scheduler = PNDMScheduler.from_pretrained(
            self.model_path, subfolder="scheduler")

        self.vae.eval()
        self.unet.eval()
        self.unet_ori.eval()

        self.model_clip, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained="/public_data/ma/models/ViT-H-14.pt")

        self.sam_model = build_sam(checkpoint=self.sam_checkpoint)
        self.sam_model.to(device=self.device)
        self.predictor = SamPredictor(self.sam_model)
        # self.cross_attention_scores = {}
        # self.unet = unet_gligen_store_cross_attention_scores(self.unet, self.cross_attention_scores, 5)
        self.image_infos = {}
        replace_clip_embeddings(self.text_encoder, self.image_infos)

    def encode_images_mask(self, user_image, user_bbox, shape):
        w, h, _ = user_image.shape
        subject_h_w = (user_bbox[3]-user_bbox[1])/(user_bbox[2]-user_bbox[0])
        bbox = torch.tensor(user_bbox) / torch.tensor([w, h, w, h])
        w1, h1 = shape[-2], shape[-1]
        bbox1 = bbox * torch.tensor([w1, h1, w1, h1])
        x1, y1 = bbox1[0], bbox1[1]
        x3, y3 = bbox1[2], bbox1[3]
        masks = []
        bboxes = []
        for i in range(shape[0]):
            mask_img = torch.ones((1, 1, h1, w1))
            chars_w, chars_h = x3-x1, y3-y1
            seed1 = random.uniform(0.8, 1.2)
            seed2 = random.uniform(0.8, 1.2)
            if chars_w < chars_h:
                chars_w, chars_h = int(
                    chars_w*seed1), int(chars_w*seed1*subject_h_w)
            else:
                chars_w, chars_h = int(
                    chars_h*seed1/subject_h_w), int(chars_h*seed1)
            chars_x, chars_y = int(x1*seed1), int(y1*seed2)
            mask_img[:, :, chars_y: chars_y + chars_h,
                     chars_x: chars_x + chars_w] = 0
            bbox = torch.tensor(([[chars_x, chars_y, chars_x + chars_w, chars_y + chars_h], [
                                0, 0, 0, 0]])) / torch.tensor([w1, h1, w1, h1])
            masks.append(mask_img)
            bboxes.append(bbox)
        return torch.cat(masks).to(self.device), torch.stack(bboxes).to(self.device)

    def encode_images_mask_custom(self,mask_bbox,shape,subject_num):
        masks = []
        bboxes = []
        for _ in range(shape[0]):
            w1 = shape[-1] * 8
            h1 = shape[-2] * 8
            mask_img = torch.ones((1, 1, w1, h1))
            if subject_num==1:
                x1, y1 = mask_bbox[0], mask_bbox[1]
                x3, y3 = mask_bbox[2], mask_bbox[3]
                chars_w, chars_h = x3-x1, y3-y1
                mask_img[:, :, y1: y1 + chars_h, x1: x1 + chars_w] = 0
                bbox = torch.tensor([[x1, y1, x1 + chars_w, y1 + chars_h], [0, 0, 0, 0]])/torch.tensor([w1, h1, w1, h1])
            else:
                bbox_double = []
                for i in range(subject_num):
                    x1, y1 = mask_bbox[i][0], mask_bbox[i][1]
                    x3, y3 = mask_bbox[i][2], mask_bbox[i][3]
                    chars_w, chars_h = x3-x1, y3-y1
                    bbox_double.append([x1, y1, x1 + chars_w, y1 + chars_h])
                    mask_img[:, :, y1: y1 + chars_h, x1: x1 + chars_w] = 0
                bbox = torch.tensor(bbox_double)/torch.tensor([w1, h1, w1, h1])
            bboxes.append(bbox)
            mask_img_resize = transforms.Resize((int(w1/8), int(h1/8)), interpolation=transforms.InterpolationMode.NEAREST)(mask_img)
            masks.append(mask_img_resize)
        return torch.cat(masks).to(self.device), torch.stack(bboxes).to(self.device)

    def encode_images_mask_random(self,images_sam,shape,subject_num):
        masks = []
        bboxes = []
        for _ in range(shape[0]):
            w1 = shape[-1] * 8
            h1 = shape[-2] * 8
            size = 350
            mask_img = torch.ones((1, 1, w1, h1))
            seed1 = random.uniform(0.6, 0.8)
            seed2 = random.uniform(0.6, 0.8)
            chars_w, chars_h = int(size*seed1), int(size*seed2)
            chars_x = random.randint(10, 200)
            chars_y = random.randint(10, 200)
            if subject_num==1:
                mask_img[:, :, chars_y: chars_y + chars_h, chars_x: chars_x + chars_w] = 0
                bbox = torch.tensor(([[chars_x, chars_y, chars_x + chars_w, chars_y + chars_h], [0, 0, 0, 0]]))/torch.tensor([w1, h1, w1, h1])
            else:
                # mask_img[:, :, 100:400, 50:200] = 0
                # mask_img[:, :, 100:400, 300:500] = 0
                # bbox = torch.tensor(([[50, 100, 200, 400],[300, 100, 500, 400]]))/torch.tensor([w1, h1, w1, h1])
                transf = transforms.Compose([transforms.Resize(
                    200, max_size=250, interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor()])
                img0 = transf(images_sam[0])
                img1 = transf(images_sam[1])
                mask_img[:,:, 100:100+img0.size(1), 20:20+img0.size(2)] = 0
                mask_img[:,:, 100:100+img1.size(1), 250:250+img1.size(2)] = 0
                bbox = torch.tensor(([[20, 100, 20+img0.size(2),100+img0.size(1)], [250, 100, 250+img1.size(2), 100+img1.size(1)]]))/torch.tensor([w1, h1, w1, h1])
            bboxes.append(bbox)
            mask_img_resize = transforms.Resize((int(w1/8), int(h1/8)), interpolation=transforms.InterpolationMode.NEAREST)(mask_img)
            masks.append(mask_img_resize)
        return torch.cat(masks).to(self.device), torch.stack(bboxes).to(self.device)

    def encode_images_test_512(self, images, bboxes):
        image_tensor = torch.empty(
            len(images), self.MAX_NUMBER_OBJECTS, 3, 224, 224, device=self.device)
        image_token_idx_mask = torch.zeros(
            len(images), self.MAX_NUMBER_OBJECTS, 1, 1, dtype=bool, device=self.device)
        for i in range(len(images)):
            if isinstance(images[i], list):
                image_token_idx_mask[i] = True
                for j in range(len(images[i])):
                    image_tensor[i][j] = self.preprocess(pad_image(images[i][j]))
            else:
                image_token_idx_mask[i, 0] = True
                image_tensor[i][0] = self.preprocess(pad_image(images[i]))
        image_embeddings_cls, image_embeddings = self.model_clip.to(
            self.device).encode_image(image_tensor.view(len(images)*self.MAX_NUMBER_OBJECTS, 3, 224, 224))
        image_embeddings_cls = self.mlp.to(self.device)(image_embeddings_cls)
        image_embeddings = rearrange(
            image_embeddings, "(b n) h d -> b n h d", n=self.MAX_NUMBER_OBJECTS)
        image_embeddings = self.proj.to(self.device)(
            image_embeddings, image_token_idx_mask, bboxes)
        return image_embeddings_cls.view(len(images), self.MAX_NUMBER_OBJECTS, -1), image_embeddings

    def encode_images_test(self, images):
        image_tensor = torch.empty(
            len(images), self.MAX_NUMBER_OBJECTS, 3, 224, 224, device=self.device)

        for i in range(len(images)):
            if isinstance(images[i], list):
                for j in range(len(images[i])):
                    image_tensor[i][j] = self.preprocess(
                        pad_image(images[i][j]))
            else:
                image_tensor[i][0] = self.preprocess(pad_image(images[i]))
        image_embeddings_cls, image_embeddings = self.model_clip.to(
            self.device).encode_image(image_tensor.view(len(images)*self.MAX_NUMBER_OBJECTS, 3, 224, 224))
        image_embeddings_cls = self.mlp.to(self.device)(image_embeddings_cls)
        image_embeddings = rearrange(
            image_embeddings, "(b n) h d -> b h (n d)", n=self.MAX_NUMBER_OBJECTS)
        image_embeddings = self.proj.to(self.device)(image_embeddings)
        return image_embeddings_cls.view(len(images), self.MAX_NUMBER_OBJECTS, -1), image_embeddings

    def generate_text_inputs(self, prompts, entities):
        image_token_mask = torch.zeros(
            (len(prompts), self.tokenizer.model_max_length), dtype=bool, device=self.device)
        image_token_idx_mask = torch.zeros(
            (len(prompts), self.MAX_NUMBER_OBJECTS), dtype=bool, device=self.device)
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
    def encode_prompt(self, prompt, entity, image, bboxes, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, token=True):
        prompts_ori = [prompt]*num_images_per_prompt
        entities = [entity]*num_images_per_prompt
        images = [image]*num_images_per_prompt
        prompts, image_token_mask, image_token_idx_mask = self.generate_text_inputs(prompts_ori, entities)
        # self.tokenizer.pad_token_id = 49407
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        if self.patch_512:
            image_embeddings_cls, image_embeddings = self.encode_images_test_512(images, bboxes)
        else:
            image_embeddings_cls, image_embeddings = self.encode_images_test(images)
        self.image_infos["image_embedding"] = image_embeddings_cls[image_token_idx_mask]
        self.image_infos["image_token_mask"] = image_token_mask

        encoder_hidden_states = self.text_encoder(text_inputs.to(self.device).input_ids)[0]

        self.image_infos["image_embedding"] = None
        self.image_infos["image_token_mask"] = None

        ori_text_input = self.tokenizer(
            prompts_ori,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder_ori(ori_text_input.to(self.device).input_ids)[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = encoder_hidden_states.shape
        # encoder_hidden_states = encoder_hidden_states.repeat(
        #     1, num_images_per_prompt, 1)
        encoder_hidden_states = encoder_hidden_states.view(
            bs_embed, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""]*num_images_per_prompt
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]*num_images_per_prompt
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

            uncond_embeddings = self.text_encoder(
                uncond_input.to(self.device).input_ids)[0]
            uncond_embeddings_ori = self.text_encoder_ori(
                uncond_input.to(self.device).input_ids)[0]
            # seq_len = uncond_embeddings.shape[1]
            # uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            # uncond_embeddings = uncond_embeddings.view(1, seq_len, -1)

            text_embeddings = torch.cat([uncond_embeddings_ori, text_embeddings])
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

    def sam(self, subject_num, images, bboxs):
        # numpy_image = np.array(image)
        if subject_num==1:
            images = [images]
            bboxs = [bboxs]
        ret_images = []
        for image, bbox in zip(images, bboxs):
            sam_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(sam_img)
            if self.inner_coords:
                w = (bbox[2]-bbox[0])//3
                h = (bbox[3]-bbox[1])//3
                point_coords = np.array([[bbox[0]+w,bbox[1]+h],[bbox[0]+2*w,bbox[1]+h],[bbox[0]+w,bbox[1]+2*h],[bbox[0]+2*w,bbox[1]+2*h]])
                point_labels = np.array([1,1,1,1])
            else:
                point_coords = None
                point_labels = None
            masks, iou_predictions, low_res_masks = self.predictor.predict(
                point_coords=point_coords, point_labels=point_labels, box=np.asarray(bbox).reshape(1, -1), multimask_output=False)
            # coord = np.array(image.size).reshape(1, 2)//2
            # masks, iou_predictions, low_res_masks = self.predictor.predict(point_coords=coord, point_labels=np.array([1]))
            # channel = self.find_main_entity(masks)
            mask = masks[0]
            mask = self.fill_cavity_2(mask)

            numpy_image = np.where(
                mask[:, :, np.newaxis].repeat(3, axis=2), sam_img, 255)
            y, x = np.where(mask)
            ret_image = Image.fromarray(numpy_image[y.min():y.max(), x.min():x.max(), :])
            ret_images.append(ret_image)
        return ret_images

    @torch.no_grad()
    def log_imgs(
        self,
        subject_num,
        user_image,
        user_bbox,
        raw_text,
        entity,
        num_images_per_prompt,
        num_inference_steps,
        height,
        width,
        mask_type=1, 
        mask_bbox=None,
        start=0,
        end=50,
        guidance_scale: float = 5,
        image_guidance_scale: float = 2.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        prompts = raw_text
        # user_image = cv2.imread(user_image)
        images_sam = self.sam(subject_num,user_image, user_bbox)
        do_classifier_free_guidance = guidance_scale > 1
        shape = (num_images_per_prompt, 4, height // 8, width // 8)

        if mask_type==0:
            latents_c2, bboxes = self.encode_images_mask_custom(mask_bbox, shape, subject_num)
        elif mask_type==1:
            latents_c2, bboxes = self.encode_images_mask(user_image, user_bbox, shape)
        else:
            latents_c2,bboxes = self.encode_images_mask_random(images_sam,shape,subject_num)
        text_embeddings, text_embeddings_ori, image_embeddings, image_embeddings_cls = self.encode_prompt(
            raw_text, entity, images_sam, bboxes, num_images_per_prompt, do_classifier_free_guidance, negative_prompt)

        self.test_scheduler.set_timesteps(
            num_inference_steps, device=self.device)
        timesteps = self.test_scheduler.timesteps

        ## 测试阶段修改
        shape = (num_images_per_prompt, 4, width // 8, height // 8)
        latents = torch.randn(shape, generator=generator,
                              device=self.device, dtype=text_embeddings.dtype)
        latents = latents * self.test_scheduler.init_noise_sigma

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        font_latents = latents_c2
        uncond_image_latents = torch.zeros_like(font_latents).to(self.device)
        font_latents = torch.cat(
            [font_latents, font_latents, uncond_image_latents], dim=0).to(self.device).half()
        # GLIGEN
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
            if start <= i <= end:
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
                # for key, value in self.cross_attention_scores.items():
                #     if key in attention_store:
                #         attention_store[key] += value
                #     else:
                #         attention_store[key] = value

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

        return image, images_sam, latents_c2

    def infer(self, inputs):
        # inputs =  json.load(inputs_paras)
        inputs_para = inputs["paras"]
        user_image = inputs_para["image"]
        user_bbox = inputs_para["bbox"]
        mask_type = inputs_para["mask_type"]
        mask_bbox = inputs_para["mask_bbox"]
        raw_text = inputs_para["text"]
        num_images_per_prompt = inputs_para["num_images"]
        num_steps = inputs_para["num_steps"]
        diversity = inputs_para["diversity"]
        height = inputs_para["width"]
        width = inputs_para["height"]
        entity = inputs_para["entity"]
        start = inputs_para["start"]
        end = inputs_para["end"]
        subject_num = inputs_para["subject_num"]
        seed = inputs_para["seed"]
        if seed!=-1:
            setup_seed(seed)

        assert not (mask_type==1 and subject_num==2), "subject_num==2时不支持mask_type==1"
        ## test
        print("mask_bbox: ",mask_bbox)
        msg = ''
        print('start:{},end:{},diversity:{}'.format(start,end,diversity))

        if subject_num==1:
            if inputs_para.__contains__('image_base64'):
                user_image = base64_to_cv2_img(inputs_para["image_base64"])
                print(type(user_image))
            # user_image = cv2.imread(user_image)
        else:
            if inputs_para.__contains__('image_base64'):
                user_image[0] = base64_to_cv2_img(inputs_para["image_base64"][0])
                user_image[1] = base64_to_cv2_img(inputs_para["image_base64"][1])
                print(type(user_image[0]))
                print(type(user_image[1]))
            # user_image = [cv2.imread(user_image[0]),cv2.imread(user_image[1])]
            
        images, images_sam, latents_c2 = self.log_imgs(
            subject_num,user_image, user_bbox, raw_text, entity, num_images_per_prompt, num_steps, height, width,mask_type, mask_bbox,start=start, end=end, guidance_scale=diversity)

        save_image(latents_c2, "subject_result/images_mask.png")
        images.extend(images_sam)
        for i, image in enumerate(images):
            images[i].save(f"subject_result/{i}.png")

        if msg == '':
            msg = '处理成功'
        result = {'status': 'SUCCESS', 'msg': msg,
                  "result_images": pil_to_base64(images)}
        return result


if __name__ == "__main__":
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        device = torch.device(f"cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device(f"cpu")

    text2Img = Text2Img(device)
    text2Img.load()
    # /public_data/liangjunhao/GLIGEN/dreambooth-main/dataset/cat2/01.jpg [305, 472, 1919, 1563] 横图
    # /public_data/liangjunhao/GLIGEN/dreambooth-main/dataset/backpack_dog/00.jpg [430, 302, 1086, 1144] 竖图
    # subject_num==1
    inputs_paras = {
        "code": 200,
        "paras": {"subject_num": 1, # subject_num=2 则"entity"， "image" ，"bbox" 为List输入
                  "text": "a backpack on the beach", 
                  "entity": "backpack",
                  "image": "/public_data/liangjunhao/GLIGEN/dreambooth-main/dataset/backpack_dog/00.jpg",
                  "bbox": [430, 302, 1086, 1144],
                  "height": 512,
                  "width": 768,
                  "num_images": 6,
                  "num_steps": 50,
                  "diversity": 7.5,
                  "mask_type": 1,  # 0表示用户自定义，需要输入mask区域,即需要指定mask_bbox; 1表示按照原图nask比例生成; 2表示随机给定mask;
                  "mask_bbox": [0, 0, 300, 300], 
                  "start": 0,
                  "end": 40,
                  "seed": 1
                  }
    }

    # subject_num==2
    # inputs_paras = {
    #     "code": 200,
    #     "paras": {"subject_num": 2, # subject_num=2 则"entity"， "image" ，"bbox" 为List输入
    #               "text": "a backpack and a cat on the beach", 
    #               "entity": ["backpack","cat"],
    #               "image": ["/public_data/liangjunhao/GLIGEN/dreambooth-main/dataset/backpack_dog/00.jpg","/public_data/liangjunhao/GLIGEN/dreambooth-main/dataset/cat2/01.jpg"],
    #               "bbox": [[430, 302, 1086, 1144],[305, 472, 1919, 1563]],
    #               "height": 512,
    #               "width": 512,
    #               "num_images": 6,
    #               "num_steps": 50,
    #               "diversity": 7.5,
    #               "mask_type": 0,  #0表示用户自定义，需要输入mask区域,即需要指定mask_bbox; subject_num==2时无法按照原比例给出，即mask_type!=1; 2表示随机给定mask;
    #               "mask_bbox": [[0, 0, 300, 300],[200,200,512,512]], 
    #               "start": 0,
    #               "end": 40
    #               }
    # }
    with open("image.txt") as f:
        image_base64 = f.readline().strip()
    inputs_paras["paras"]["image_base64"] = image_base64
    result = text2Img.infer(inputs_paras)
    # print('text2Img.infer :'+str(result))
