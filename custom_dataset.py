import io
import random
import cv2
from torchvision.transforms import functional as F
import braceexpand
from torchvision.utils import save_image
from transformers import CLIPTokenizer
import os
import numpy as np
import torch
import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import imgviz
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from prefetch_generator import BackgroundGenerator
from pytorch_lightning import LightningDataModule

colormap = imgviz.label_colormap(80)
USED_KEYS = {"jpg": "instance_images",
             "json": "mask", "png": "instance_masks"}
max_num_objects = 2
area_min = 0.08
area_max = 0.7
ratio_min = 0.3
ratio_max = 3
score = 0.3
iou_ratio = 0.8
fill_bbox_ratio = 0.6
max_bbox_num_subj = 5


SKS_ID = 48136


def expand_urls(urls):
    if isinstance(urls, str):
        urllist = urls.split("::")
        result = []
        for url in urllist:
            result.extend(braceexpand.braceexpand(url))
        return result
    else:
        return list(urls)


def custom_decoder(key, data):
    "Customize the decoder to process the original image and mask, and the original image processing method is equivalent to pil or pilrgb"
    import PIL.Image
    if key.endswith("png"):
        # return None
        with io.BytesIO(data) as stream:
            img = PIL.Image.open(stream)
            img.load()
            # result = img.convert("RGB")
            # result = img.convert("L")
            return img
    elif key.endswith("jpg"):
        with io.BytesIO(data) as stream:
            img = PIL.Image.open(stream)
            img.load()
            result = img.convert("RGB")
            return result
    else:
        return None


def is_valid_bbox(mask, bbox):
    crop_mask = mask[0, int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    colors, mask_area = crop_mask.unique(return_counts=True)
    # Limit the number of entities within a bbox
    if len(mask_area) > max_bbox_num_subj or len(mask_area) < 2:
        return False
    # The proportion of the main mask area within the box is greater than the threshold
    if mask_area.max()/(bbox[2]-bbox[0])/(bbox[3]-bbox[1]) < fill_bbox_ratio:
        return False
    # Filter out situations where the mask area of the main body is larger than that of the bbox
    if (mask == colors[mask_area[1:].argmax()+1]).sum() > (bbox[2]-bbox[0])*(bbox[3]-bbox[1]):
        return False
    return True


def is_contained(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 > x1 and y2 > y1:
        ratio = (y2-y1)*(x2-x1)/min((box1[2]-box1[0]) *
                                    (box1[3]-box1[1]), (box2[2]-box2[0])*(box2[3]-box2[1]))
        if ratio > iou_ratio:
            return True
    return False


def verify_keys(samples, required_keys, hr_size, handler=wds.handlers.reraise_exception):
    """
    Requires that both the image and embedding are present in the sample
    This is important to do as a user may forget they do not have embeddings in their webdataset and neglect to add them using the embedding_folder_url parameter.
    """
    for sample in samples:
        try:
            w, h = sample["jpg"].size
            mask = transforms.ToTensor()(sample["png"])
            image = transforms.ToTensor()(sample["jpg"])
            # Filter images with a high proportion of background and more white color
            if torch.all((image > 0.9), dim=0).sum() > w*h/10:
                continue

            if "mask" in sample["json"]:
                cat_prompts = [name["label"]
                               for name in sample["json"]["mask"] if "box" in name]
                bbox_ori = torch.stack(
                    [torch.tensor(name["box"]) for name in sample["json"]["mask"] if "box" in name])
                w_h_ratio = (bbox_ori[:, 2]-bbox_ori[:, 0]) / \
                    (bbox_ori[:, 3]-bbox_ori[:, 1])
                area_ratio = (bbox_ori[:, 2]-bbox_ori[:, 0]) * \
                    (bbox_ori[:, 3]-bbox_ori[:, 1])/w/h
                # Priority selection for high scoring bbox
                # bbox = bbox_ori / torch.tensor([w,h,w,h]) # 归一化
                # bbox_area = [float(abs((x[2]-x[0])*(x[3]-x[1]))) for x in bbox]
                logits = torch.tensor([t["logit"]
                                      for t in sample['json']["mask"][1:]])
                indices = logits.argsort(descending=True)

                # bbox_w_h = [float(abs((x[2]-x[0])-(x[3]-x[1]))) for x in bbox]
                bbox_selects = []
                for index in indices:
                    if area_min < area_ratio[index] < area_max and ratio_min < w_h_ratio[index] < ratio_max and logits[index] > score and " " not in cat_prompts[index] and sample["json"]["caption"].find(cat_prompts[index]) != -1:
                        # if area_min<i<area_max and j<w_h and s>score and sample["json"]["mask"][indexs+1]["label"] in sample["json"]["capFtion"]:
                        if not is_valid_bbox(mask, bbox_ori[index]):
                            continue
                        # Filter high iou and duplicate entities
                        flag = True
                        for bbox_select in bbox_selects:
                            if is_contained(bbox_ori[bbox_select], bbox_ori[index]):
                                flag = False
                                break
                        if flag:
                            bbox_selects.append(index)

                if len(bbox_selects) > 0:
                    ret_val = {key: sample[key] for key in required_keys}
                    ret_val["bbox_selects"] = bbox_selects
                    yield ret_val
            # coco
            else:
                if len(sample['json']["bboxes"]) > 0:
                    bbox_ori = torch.tensor(sample["json"]["bboxes"])
                    w_h_ratio = (bbox_ori[:, 2]-bbox_ori[:, 0]) / \
                        (bbox_ori[:, 3]-bbox_ori[:, 1])

                    area_ratio = (bbox_ori[:, 2]-bbox_ori[:, 0]) * \
                        (bbox_ori[:, 3]-bbox_ori[:, 1])/w/h
                    indices = area_ratio.argsort(descending=True)
                    bbox_selects = []
                    cat_prompts = sample["json"]['cat_names']
                    for index in indices:
                        if area_min < area_ratio[index] < area_max and ratio_min < w_h_ratio[index] < ratio_max and is_valid_bbox(mask, bbox_ori[index]):
                            flag = True
                            for bbox_select in bbox_selects:
                                if is_contained(bbox_ori[bbox_select], bbox_ori[index]):
                                    flag = False
                                    break
                            if flag:
                                bbox_selects.append(index)
                    if len(bbox_selects) > 0:
                        # ret_val = {key: sample[key] for key in required_keys}
                        sample["bbox_selects"] = bbox_selects
                        yield sample
        except Exception as exn:  # From wds implementation
            if handler(exn):
                continue
            else:
                break


def fill_cavity(input_mask):
    # fast but not accurate
    cumsum = input_mask.cumsum(-1)
    filled_mask = (cumsum > 0)
    filled_mask &= (cumsum < cumsum[..., -1:])
    cumsum = input_mask.cumsum(-2)
    filled_mask &= (cumsum > 0)
    filled_mask &= (cumsum < cumsum[..., -1:, :])
    return filled_mask


def image_seg(bbox, pixel_seg):

    cropped_seg = pixel_seg[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]

    colors, mask_counts = torch.unique(
        cropped_seg, return_counts=True, sorted=True)
    if len(mask_counts) > 1:
        max_mask_idx = mask_counts[1:].argmax()+1
    elif len(mask_counts) == 1:
        max_mask_idx = 0
    else:
        return torch.zeros_like(pixel_seg)
    return fill_cavity((pixel_seg == colors[max_mask_idx]).float())


def post_bbox_filter(bbox):
    if (bbox[3]-bbox[1]) * (bbox[2]-bbox[0]) > 200 and ratio_min < (bbox[3]-bbox[1])/(bbox[2]-bbox[0]) < ratio_max:
        return True
    else:
        return False


def make_prompt(tokenizer, ori_prompt, entities):
    template_text = ori_prompt.strip() + \
        "".join([f", the {entity} is sks" for entity in entities])
    input_ids = tokenizer(
        template_text,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    image_token_mask = (input_ids == SKS_ID)[0]
    image_token_idx = image_token_mask.nonzero().squeeze(1)
    if len(image_token_idx) < max_num_objects:
        image_token_idx = torch.cat((image_token_idx, torch.zeros(
            max_num_objects-len(image_token_idx), dtype=bool)))
    elif len(image_token_idx) > max_num_objects:
        image_token_idx = image_token_idx[-max_num_objects:]
    return template_text, input_ids, image_token_mask, image_token_idx


def get_entity_image(bbox, object_segmap, instance_image, pad_white=1):
    image_augmentation = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=10), transforms.RandomPerspective(distortion_scale=0.5, p=0.5), transforms.Resize(
        224, interpolation=transforms.functional._interpolation_modes_from_int(0)),
        transforms.RandomCrop(224), transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(
            0.26862954, 0.26130258, 0.27577711))])
    if pad_white:
        image = (instance_image*object_segmap+1 -
                 object_segmap)[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]
    else:
        image = (instance_image *
                 object_segmap)[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]
    if bbox[3]-bbox[1] < bbox[2]-bbox[0]:
        pad_size = int(bbox[2]-bbox[0]-bbox[3]+bbox[1])//2
        image = F.pad(image, (0, pad_size, 0, pad_size), pad_white)
    elif bbox[3]-bbox[1] > bbox[2]-bbox[0]:
        pad_size = int(-bbox[2]+bbox[0]+bbox[3]-bbox[1])//2
        image = F.pad(image, (pad_size, 0, pad_size, 0), pad_white)
    return image_augmentation(image)


def post_verify(samples, tokenizer, hr_size, handler=wds.handlers.reraise_exception):
    """
    Requires that both the image and embedding are present in the sample
    This is important to do as a user may forget they do not have embeddings in their webdataset and neglect to add them using the embedding_folder_url parameter.
    """
    for sample in samples:
        masks = sample["mask"]
        input_ids = tokenizer(
            sample["instance_prompt"],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        if (input_ids == SKS_ID).any():
            continue
        kept_masks = []
        kept_entities = []
        bboxes = torch.zeros((max_num_objects, 4))
        image_token_idx_mask = torch.zeros((max_num_objects), dtype=bool)
        padded_object_segmaps = torch.zeros(
            (max_num_objects, hr_size, hr_size))
        entity_images = torch.zeros(max_num_objects, 3, 224, 224)
        for i, mask in enumerate(masks):
            y, x = torch.where(mask == 0)

            if len(y) == 0 or not post_bbox_filter([x.min(), y.min(), x.max(), y.max()]):
                continue
            bbox = (x.min(), y.min(), x.max(), y.max())
            padded_object_segmaps[len(kept_masks)] = image_seg(
                bbox, sample["instance_seg"])
            entity_images[len(kept_masks)] = get_entity_image(
                bbox, padded_object_segmaps[len(kept_masks)], sample["instance_image"])
            image_token_idx_mask[len(kept_masks)] = True
            bboxes[len(kept_masks)] = torch.tensor(
                [bbox])/512
            kept_masks.append(mask.unsqueeze(0))
            kept_entities.append(sample["cat_prompts"][i])

            if len(kept_masks) == max_num_objects:
                break
        if len(kept_masks) == 0:
            continue
        else:
            sample["mask"] = torch.cat(kept_masks)
            sample["cat_prompts"] = kept_entities
            sample["bboxes"] = bboxes

            template_text, input_ids, image_token_mask, image_token_idx = make_prompt(tokenizer,
                                                                                      sample["instance_prompt"], kept_entities)

            sample["instance_prompt"] = template_text
            sample["image_token_idx"] = image_token_idx
            # b, max_num_objects, _, _
            sample["object_segmaps"] = padded_object_segmaps
            # if image_token_idx[0] == image_token_idx[1]:
            #     image_token_idx[1] += 1
            # b, max_num_objects
            sample["image_token_idx_mask"] = image_token_idx_mask
            sample["entity_images"] = entity_images

            # sample["input_ids"] = input_ids
            sample["image_token_mask"] = image_token_mask
            yield sample


key_verifier = wds.filters.pipelinefilter(verify_keys)
post_key_verifier = wds.filters.pipelinefilter(post_verify)


class ImageEmbeddingDataset(wds.DataPipeline, wds.compat.FluidInterface):
    """
    A fluid interface wrapper for DataPipline that returns image embedding pairs
    Reads embeddings as npy files from the webdataset if they exist. If embedding_folder_url is set, they will be inserted in from the alternate source.
    """

    def __init__(
            self,
            urls,
            tokenizer=None,
            extra_keys=[],
            hr_size=-1,
            size=512,
            handler=wds.handlers.reraise_exception,
            resample=False,
            shuffle_shards=True,
            center_crop=False,
            shuffle_cat=False,
            replace_bg=True
    ):
        super().__init__()
        keys = list(USED_KEYS.keys()) + extra_keys
        # self.key_map = {key: i for i, key in enumerate(keys)}
        self.resampling = resample
        self.hr_size = hr_size
        self.size = size
        self.shuffle_cat = shuffle_cat

        self.tokenizer = tokenizer

        if resample:
            assert not shuffle_shards, "Cannot both resample and shuffle"
            self.append(wds.ResampledShards(urls))
        else:
            self.append(wds.SimpleShardList(urls))
            if shuffle_shards:
                self.append(wds.filters.shuffle(1000))

        self.append(wds.tarfile_to_samples(handler=handler))

        self.append(wds.decode(custom_decoder, handler=handler))

        self.append(key_verifier(required_keys=keys,
                    hr_size=hr_size, handler=handler))
        # Apply preprocessing
        self.append(wds.map(self.preproc))
        self.append(post_key_verifier(tokenizer=tokenizer,
                    hr_size=hr_size, handler=handler))
        self.image_transforms_mask = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.functional._interpolation_modes_from_int(0)),
                transforms.CenterCrop(size),
                transforms.ToTensor()
            ]
        )

        self.image_transforms_mask_nocrop = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.functional._interpolation_modes_from_int(0)),
                transforms.ToTensor(),
            ]
        )

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                # transforms.Normalize(0.5,0.5)
            ]
        )

        self.image_transforms_nocrop = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                # transforms.Normalize(0.5,0.5)
            ]
        )
        self.replace_bg = replace_bg
        if replace_bg:
            self.bg_transform = transforms.Compose(
                [transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.RandomCrop(224),

                    transforms.ToTensor(),
                    # transforms.RandomHorizontalFlip(0.5)
                 ]
            )
            self.bg_img_path = "/data_share/liangjunhao/BG-20k/train"
            self.bg_img_list = os.listdir(self.bg_img_path)
            self.append(wds.map(self.synthesize))
            self.image_augmentation = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=10), transforms.RandomPerspective(distortion_scale=0.5, p=0.5), transforms.Resize(
                224, interpolation=transforms.functional._interpolation_modes_from_int(0)),
                transforms.RandomCrop(224), transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(
                    0.26862954, 0.26130258, 0.27577711))])

    def transform(self, image_ori, bbox, bbox_selects, image_seg):
        # custom resize
        mask_imgs = []
        mask_imgs_crop = []
        w, h = image_ori.width, image_ori.height
        min_x, min_y, max_x, max_y = int(bbox[:, 0].min()), int(
            bbox[:, 1].min()), int(bbox[:, 2].max()), int(bbox[:, 3].max())
        for bbox_select in bbox_selects:
            mask_img = np.zeros((h, w))
            x_1, y_1, x_2, y_2 = bbox[bbox_select]
            polygon = np.array([[x_1, y_1], [x_2, y_1], [x_2, y_2], [
                               x_1, y_2]], np.int32) 
            mask_img = cv2.fillConvexPoly(mask_img, polygon, (1, 1, 1))
            mask_img = Image.fromarray(mask_img)
            mask_imgs.append(mask_img)

        # Using the largest box as the drop benchmark
        # x_1,y_1,x_2,y_2 = bbox[bbox_selects[0]]
        # polygon = np.array([[x_1,y_1],[x_2,y_1],[x_2,y_2],[x_1,y_2]], np.int32) # 坐标为顺时针方向
        crop_size = min(w, h)

        # x1, x2, y1, y2 = np.min(polygon[:,0]),np.max(polygon[:,0]),np.min(polygon[:,1]),np.max(polygon[:,1])
        x_b, x_e = max(0, max_x - crop_size), min(min_x, w - crop_size)
        y_b, y_e = max(0, max_y - crop_size), min(min_y, h - crop_size)
        # bbox_selects_new = []
        if x_b <= x_e and y_b <= y_e:
            start_x = random.randint(
                max(0, max_x - crop_size), min(min_x, w - crop_size))
            start_y = random.randint(
                max(0, max_y - crop_size), min(min_y, h - crop_size))
            instance_image_crop = F.crop(
                image_ori, start_y, start_x, crop_size, crop_size)
            instance_image_seg_crop = F.crop(
                image_seg, start_y, start_x, crop_size, crop_size)
            image = self.image_transforms_nocrop(instance_image_crop)
            image_seg = self.image_transforms_mask_nocrop(
                instance_image_seg_crop)
            for i, mask_img in enumerate(mask_imgs):
                mask_img = F.crop(mask_img, start_y, start_x,
                                  crop_size, crop_size)
                mask_img = self.image_transforms_mask_nocrop(mask_img)
                # bbox_selects_new.append(bbox_selects[i])
                mask_imgs_crop.append(1-mask_img)
                # else:
                #     if len(torch.unique(mask_img))!=1:
                #         bbox_selects_new.append(bbox_selects[i])
                #         mask_imgs_crop.append(1-mask_img)

        else:
            for i, mask_img in enumerate(mask_imgs):
                mask_img = self.image_transforms_mask(mask_img)
                # if i==0:
                # bbox_selects_new.append(bbox_selects[i])
                mask_imgs_crop.append(1-mask_img)
                # else:
                # if len(torch.unique(mask_img))!=1:
                # bbox_selects_new.append(bbox_selects[i])
                # mask_imgs_crop.append(1-mask_img)
            image = self.image_transforms(image_ori)
            image_seg = self.image_transforms_mask(image_seg)

        return image, torch.cat(mask_imgs_crop), image_seg
        # return image, torch.prod(torch.cat(mask_imgs_crop),dim=0), image_seg

    def preproc(self, sample):
        example = {}
        instance_image = sample["jpg"]
        w, h = instance_image.size
        # SAM data
        if "mask" in sample["json"]:
            bbox_ori = torch.stack([torch.tensor(name["box"])
                                   for name in sample["json"]["mask"] if "box" in name])

            bbox_selects = sample["bbox_selects"]

            if not instance_image.mode == "RGB":
                instance_image = instance_image.convert("RGB")

            example["instance_image"], example["mask"], example["instance_seg"] = self.transform(
                instance_image, bbox_ori, bbox_selects, sample["png"])
            example["instance_prompt"] = sample["json"]["caption"]
            # example["bbox"] = [bbox[bbox_select] for bbox_select in bbox_selects]
            example["cat_prompts"] = [[name["label"] for name in sample["json"]
                                       ["mask"] if "box" in name][bbox_select] for bbox_select in bbox_selects]

        # coco data
        else:
            bbox_ori = torch.tensor(sample["json"]["bboxes"])
            # bbox = bbox_ori / torch.tensor([w, h, w, h])  # 归一化
            # bbox_area = [float(abs((x[2]-x[0])*(x[3]-x[1]))) for x in bbox]
            # bbox_w_h = [float(abs((x[2]-x[0])-(x[3]-x[1]))) for x in bbox]
            bbox_selects = sample["bbox_selects"]
            example["instance_image"], example["mask"], example["instance_seg"] = self.transform(
                instance_image, bbox_ori, bbox_selects, sample["png"])
            # example["bbox"] = [bbox[bbox_select] for bbox_select in bbox_selects]
            example["cat_prompts"] = [sample["json"]["cat_names"]
                                      [bbox_select] for bbox_select in bbox_selects]
            if "txt" in sample:
                example["instance_prompt"] = sample["txt"]
            else:
                example["instance_prompt"] = sample["json"]["caption"]

            # example["bbox"] = bbox_ori[bbox_selects]
            # example["bbox_num"] = len(sample["json"]["bboxes"])

        return example

    def synthesize(self, sample):

        def get_matting_coords(h, w, H=224, W=224):
            scale = (H*W/2/h/w)**0.5
            h = min(int(h*scale), H)
            w = min(int(w*scale), W)
            x1 = random.randint(0, W-w)
            y1 = random.randint(0, H-h)
            return x1, y1, w, h
        bg_imgs = torch.zeros(max_num_objects, 3, 224, 224)
        for i, bbox in enumerate(sample["bboxes"]):
            if not sample["image_token_idx_mask"][i]:
                continue
            bbox = (bbox*512).long()
            bg_path = random.choice(self.bg_img_list)
            bg_img = Image.open(os.path.join(self.bg_img_path, bg_path))
            bg_img = self.bg_transform(bg_img)
            mask = sample["object_segmaps"][i:i+1,
                                            bbox[1]:bbox[3], bbox[0]:bbox[2]]
            img = sample["instance_image"][:, bbox[1]:bbox[3], bbox[0]:bbox[2]]

            x1, y1, w, h = get_matting_coords(mask.size(-2), mask.size(-1))
            mask = transforms.Resize(size=(h, w))(mask)
            img = transforms.Resize(size=(h, w))(img)
            bg_img[:, y1:y1+h, x1:x1+w] = bg_img[:,
                                                 y1:y1+h, x1:x1+w]*(1-mask)+img*mask
            bg_imgs[i] = self.image_augmentation(bg_img)

        sample["entity_images"] = bg_imgs
        return sample


def collate_fn(examples):
    texts = []
    pixel_values = []
    masks = []
    entitys = []
    pixel_segs = []
    object_segmaps = []
    image_token_idx = []
    image_token_idx_mask = []
    entity_images = []
    bboxes = []
    image_token_mask = []

    for example in examples:
        texts.append(example["instance_prompt"])
        pixel_values.append(example["instance_image"])
        pixel_segs.append(example["instance_seg"])
        entitys.append(example["cat_prompts"])
        masks.append(example["mask"])
        object_segmaps.append(example["object_segmaps"])
        image_token_idx.append(example["image_token_idx"])
        image_token_idx_mask.append(example["image_token_idx_mask"])
        entity_images.append(example["entity_images"])
        bboxes.append(example["bboxes"])
        image_token_mask.append(example["image_token_mask"])
    # masks = torch.stack(masks)
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(
        memory_format=torch.contiguous_format).float()
    pixel_segs = torch.stack(pixel_segs)
    pixel_segs = pixel_segs.to(
        memory_format=torch.contiguous_format).float()
    batch = {
        "texts": texts,
        "pixel_values": pixel_values,
        "pixel_segs": pixel_segs,
        "entitys": entitys,
        "masks": masks,
        "object_segmaps": torch.stack(object_segmaps).to(
            memory_format=torch.contiguous_format),
        "image_token_idx": torch.stack(image_token_idx).to(
            memory_format=torch.contiguous_format),
        "image_token_idx_mask": torch.stack(image_token_idx_mask).to(
            memory_format=torch.contiguous_format),
        "image_token_mask": torch.stack(image_token_mask).to(
            memory_format=torch.contiguous_format),
        "entity_images": torch.stack(entity_images).to(memory_format=torch.contiguous_format),
        "bboxes": torch.stack(bboxes).to(memory_format=torch.contiguous_format),
    }

    return batch


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class DataModuleCustom(LightningDataModule):
    @ staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('Universal DataModule')
        parser.add_argument('--webdataset_base_urls', type=str, nargs="+")
        parser.add_argument('--num_workers', default=2, type=int)
        parser.add_argument('--batch_size', default=16, type=int)
        # parser.add_argument('--start_shard', default=0, type=int)
        # parser.add_argument('--end_shard', default=1000, type=int)
        parser.add_argument('--shard_width', default=5, type=int)
        parser.add_argument('--hr_size', default=-1, type=int)
        parser.add_argument('--train_split', default=1.0, type=float)
        parser.add_argument('--val_split', default=0.0, type=float)
        parser.add_argument('--test_split', default=0.0, type=float)
        parser.add_argument('--shuffle_train',
                            default=False, action="store_true")
        parser.add_argument('--resample_train',
                            default=False, action="store_true")
        parser.add_argument('--shuffle_num', default=None, type=int)
        parser.add_argument('--test_prompts', type=str,
                            default="./test_prompts.json")
        parser.add_argument('--test_repeat', default=1, type=int)
        parser.add_argument('--shuffle_cat', default=False,
                            action="store_true")

        parser.add_argument(
            "--resolution", type=int, default=512,
            help=(
                "The resolution for input images, all the images in the train/validation dataset will be resized to this"
                " resolution"
            ),
        )
        parser.add_argument(
            "--center_crop", action="store_true", default=False,
            help="Whether to center crop images before resizing to resolution"
        )
        return parent_args

    def __init__(
        self,
        args,
        tokenizer=None,
        collate_fn=None,
        use_worker_init_fn=None,
    ):
        super().__init__()
        # self.available_shards = list(range(args.start_shard, args.end_shard + 1))
        # if splits is None:
        #     splits = []
        splits = {
            'train': args.train_split,
            'val': args.val_split,
            'test': args.test_split,
        }
        self.webdataset_base_urls = args.webdataset_base_urls
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.shuffle_train = args.shuffle_train
        self.resample_train = args.resample_train
        self.shard_width = args.shard_width
        self.hr_size = args.hr_size
        self.use_worker_init_fn = use_worker_init_fn
        self.shuffle_num = args.shuffle_num
        self.tokenizer = tokenizer
        self.collate_fn = collate_fn
        self.center_crop = args.center_crop
        self.resolution = args.resolution
        self.shuffle_cat = args.shuffle_cat,

        self.train_prop = self.val_prop = self.test_prop = 0
        self.datasets = {}
        if splits['train'] > 0:
            self.train_prop = splits['train']
            self.train_dataloader = self._train_dataloader
            self.datasets['train'] = None
        if splits['val'] > 0:
            self.val_prop = splits['val']
            self.val_dataloader = self._val_dataloader
            self.datasets['val'] = None
        if splits['test'] > 0:
            self.test_prop = splits['test']
            self.test_dataloader = self._test_dataloader
            self.datasets['test'] = None

        self.prepare_data()
        self.setup()

    def prepare_data(self):
        assert self.train_prop + self.test_prop + self.val_prop == 1

        all_urls = []
        for url in self.webdataset_base_urls:
            all_urls += expand_urls(url)
        num_train = round(self.train_prop*len(all_urls))
        num_test = round(self.test_prop*len(all_urls))
        num_val = len(all_urls) - num_train - num_test
        assert num_train + num_test + \
            num_val == len(
                all_urls), f"{num_train} + {num_test} + {num_val} = {num_train + num_test + num_val} != {len(all_urls)}"
        self.train_urls, self.test_urls, self.val_urls = random_split(
            all_urls, [num_train, num_test, num_val])  # , generator=torch.Generator().manual_seed(self.seed)

    def setup(self, stage=None):
        if 'train' in self.datasets:
            self.datasets['train'] = ImageEmbeddingDataset(
                self.train_urls,
                self.tokenizer,
                shuffle_shards=self.shuffle_train,
                resample=self.resample_train,
                hr_size=self.hr_size,
                handler=wds.handlers.warn_and_continue,
                center_crop=self.center_crop,
                size=self.resolution,
                shuffle_cat=self.shuffle_cat,
            )
            if self.shuffle_num is not None and self.shuffle_num > 0:
                self.datasets['train'].shuffle(self.shuffle_num)
        if 'val' in self.datasets:
            self.datasets['val'] = ImageEmbeddingDataset(
                self.val_urls,
                self.tokenizer,
                shuffle_shards=False,
                resample=False,
                hr_size=self.hr_size,
                handler=wds.handlers.warn_and_continue,
                center_crop=self.center_crop,
                size=self.resolution,
                shuffle_cat=self.shuffle_cat,
            )
        if 'test' in self.datasets:
            self.datasets['test'] = ImageEmbeddingDataset(
                self.test_urls,
                self.tokenizer,
                shuffle_shards=False,
                resample=False,
                hr_size=self.hr_size,
                handler=wds.handlers.warn_and_continue,
                center_crop=self.center_crop,
                size=self.resolution,
                shuffle_cat=self.shuffle_cat,
            )

    def _train_dataloader(self):
        # return self.create_dataloader(self.train_urls, shuffle=self.shuffle_train, resample=self.resample_train)
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoaderX(
            self.datasets['train'],
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            prefetch_factor=2,  # This might be good to have high so the next npy file is prefetched
            pin_memory=True,
            shuffle=False,
            worker_init_fn=init_fn,
            collate_fn=self.collate_fn,
        )

    def _val_dataloader(self, shuffle=False):
        # return self.create_dataloader(self.val_urls, shuffle=False)
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoaderX(
            self.datasets['val'],
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            prefetch_factor=2,  # This might be good to have high so the next npy file is prefetched
            pin_memory=True,
            shuffle=False,
            worker_init_fn=init_fn,
            collate_fn=self.collate_fn,
        )

    def _test_dataloader(self, shuffle=False):
        # return self.create_dataloader(self.test_urls, shuffle=False)
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoaderX(
            self.datasets['test'],
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            prefetch_factor=2,  # This might be good to have high so the next npy file is prefetched
            pin_memory=True,
            shuffle=False,
            worker_init_fn=init_fn,
            collate_fn=self.collate_fn,
        )


if __name__ == '__main__':
    url = "/public_data/ma/data_process/aesthetics_tar_sam/{}.tar"
    available_shards = list(range(120, 121))
    tokenizer = CLIPTokenizer.from_pretrained(
        "/public_data/wrc/models/stable-diffusion-v1-5", subfolder="tokenizer")
    # available_shards = list(range(1919,1920))
    urls = [url.format(str(shard).zfill(5)) for shard in available_shards]
    ds = ImageEmbeddingDataset(
        urls,
        shuffle_shards=True,
        resample=False,
        hr_size=512,
        tokenizer=tokenizer,
        handler=wds.handlers.warn_and_continue
    )

    loader = DataLoaderX(
        ds,
        num_workers=1,
        batch_size=1,
        prefetch_factor=2,  # This might be good to have high so the next npy file is prefetched
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn
    )
    cnt = []
    from collections import Counter
    for i, batch in tqdm(enumerate(loader)):
        masks = batch["masks"][0]
        cnt.append(len(masks))
    print(Counter(cnt))
