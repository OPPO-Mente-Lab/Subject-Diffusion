import os,sys
import json
from time import time
from tqdm import tqdm
import webdataset as wds

import torch
from PIL import Image
from torchvision.utils import save_image

import torchvision
from PIL import Image

from torchvision import transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2

# BLIP
from transformers import BlipProcessor, BlipForConditionalGeneration,Blip2Processor, Blip2ForConditionalGeneration

import spacy
nlp = spacy.load("en_core_web_sm")
import imgviz
colormap = imgviz.label_colormap(80)


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path)
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    # print(load_res)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold,device):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases

def check_caption_spacy(caption, pred_phrases,caption_lemma):
    num2en = {2: "two",3: "three",4: "four",5: "five",6: "six"}
    object_list = [obj.split('(')[0] for obj in pred_phrases]
    caption_list = caption.split(" ")
    for obj in set(object_list):
        nums = object_list.count(obj)
        if 1<nums<7:
            nums_en = num2en[nums]
            word = nlp(obj)[0].lemma_
            if word not in caption_lemma:continue
            index = caption_lemma.index(word)
            caption_list.insert(index,nums_en)
    return " ".join(caption_list)

def image_tensor2cv2(input_tensor: torch.Tensor):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # RGB转BRG
    image_cv = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(input_tensor)

    return input_tensor

def image_tensor2pillow(input_tensor: torch.Tensor):

    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # input_tensor = input_tensor
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为numpy
    # input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).type(torch.uint8).numpy()
    # 转成pillow
    lbl_pil = Image.fromarray(input_tensor.type(torch.uint8).numpy(),mode='P')
    lbl_pil.putpalette(colormap.flatten())
    return lbl_pil

class BLIP:
    def __init__(self,device):
        # self.processor = Blip2Processor.from_pretrained("/data_share/zhaomingjun/data_cleaning/blip2-opt-2.7b")
        # self.model = Blip2ForConditionalGeneration.from_pretrained("/data_share/zhaomingjun/data_cleaning/blip2-opt-2.7b", torch_dtype=torch.float16)
        self.processor = BlipProcessor.from_pretrained("/public_data/ma/models/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("/public_data/ma/models/blip-image-captioning-large", torch_dtype=torch.float16)
        self.model.to(device)
        self.model.eval()

    def get_caption(self, image):
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.model.device, torch.float16)
            generated_ids = self.model.generate(**inputs)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_text

class Data_pro:
    def __init__(self,device):
        # blip
        self.blip = BLIP(device)

        # grounding dino model
        grounded_checkpoint = "groundingdino_swint_ogc.pth"
        self.config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.grounding_model = load_model(self.config_file, grounded_checkpoint, device=device)
        self.grounding_model.to(device=device)

        # sam model
        sam_checkpoint = "/public_data/ma/models/sam/sam_vit_h_4b8939.pth"
        self.sam_model = build_sam(checkpoint=sam_checkpoint)
        self.sam_model.to(device=device)
        self.predictor = SamPredictor(self.sam_model)

        self.box_threshold = 0.25
        self.text_threshold = 0.2
        self.iou_threshold = 0.5


        self.image_transforms = T.Compose([
            T.RandomResize([800], max_size=3000),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.image_transforms_save = T.Compose([
            T.RandomResize([800], max_size=3000)])

        self.key_verifier = wds.filters.pipelinefilter(self.verify_keys)
        self.device = device
        

    def normalized(self,a, axis=-1, order=2):
        import numpy as np  # pylint: disable=import-outside-toplevel

        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)

    def get_count(self,input_file):
        stats_file = input_file[:-4] + "_stats.json"
        f = open(stats_file)
        stats = json.load(f)
        f.close()
        count = stats["successes"]
        return count

    def preproc(self, sample):
   
        instance_image = sample["jpg"]
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        # instance_image.save("1.png")

        sample["image"] = instance_image
        # sample["image"],_ = self.image_transforms(instance_image,None)

        return sample
                
    def verify_keys(self,samples,required_keys,hr_size=600):
        """
        Requires that both the image and embedding are present in the sample
        This is important to do as a user may forget they do not have embeddings in their webdataset and neglect to add them using the embedding_folder_url parameter.
        """
        for sample in samples:
            for key in required_keys:
                assert key in sample, f"Sample {sample['__key__']} missing {key}. Has keys {sample.keys()}"
            if sample['json']['original_width'] >= hr_size or sample['json']['original_height'] >= hr_size:
                yield {key:sample[key] for key in required_keys}

    def filter_dataset(self,item):
        meta = item["json"]
        # meta['caption']=zhconv.convert(re.sub(r'[^\u4E00-\u9FA5,.!?:;，。！？：；1234567890]', '', meta['caption'][:64]), 'zh-hans')
        if meta['original_width'] < 224 or meta['original_height'] < 224:
            return False
        # if len(meta['caption']) < 5:
        #     return False
        return True


    def shuffle_augment_wds(self,input, output):
        start = time()
        # count = get_count(input)
        input = "file:"+input
        pre_name = os.path.split(input)[-1][:2]
        src = wds.DataPipeline(
            wds.SimpleShardList(input),
            wds.tarfile_to_samples(),
            wds.decode("pil"),
            self.key_verifier(required_keys=["__key__", "jpg", "txt","json"]),
            # wds.select(self.filter_dataset),
            wds.map(self.preproc),
            wds.to_tuple("__key__", "jpg", "txt","json","image"),
            wds.batched(200)
        )
        
        samples = []
        # 考虑两个边界：1 分辨率全部过滤 2 美学评分全部过滤
        for i,(keys, _, cap_oris,json,images) in enumerate(tqdm(src, desc=f"Extracting {input}")):
            # if i>20:continue
            # 生成描述
            captions = self.blip.get_caption(images)

            caption_lemmas = []   
            tag_prompts = []       

            text_prompt_list = []
            caption_lemma = []
            doc = nlp("! ".join(captions)+"!")
            for token in doc:
                if token.text!="!":
                    caption_lemma.append(token.lemma_)
                    if token.pos_=="NOUN":
                        text_prompt_list.append(str(token))
                else:
                    tag_prompt = ",".join(text_prompt_list)
                    tag_prompts.append(tag_prompt)
                    caption_lemmas.append(caption_lemma)
                    text_prompt_list = []
                    caption_lemma = []


            # 生成检测框(考虑到不改变原始图片形状，one by one)
            json_datas = []
            mask_imgs = []
            cap_oris_new = []
            keys_new = []
            images_new = []
            for img,tag_prompt,caption_lemma,caption,cap_ori,key in zip(images,tag_prompts,caption_lemmas,captions,cap_oris,keys):
                # 过滤掉只包含两个实体
                if len(tag_prompt.split(","))<3 or len(tag_prompt.split(","))>8:
                    continue
                image_save = self.image_transforms_save(img,None)[0]
                image = self.image_transforms(img,None)[0]
                boxes_filt, scores, pred_phrases = get_grounding_output(self.grounding_model, image, tag_prompt, self.box_threshold, self.text_threshold, device=self.device)
                if len(pred_phrases)<2 or len(pred_phrases)>8:
                    continue
                size = image.shape
                H, W = size[-2], size[-1]
                for i in range(boxes_filt.size(0)):
                    boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H]).to(self.device)
                    boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                    boxes_filt[i][2:] += boxes_filt[i][:2]

                # use NMS to handle overlapped boxes
                # print(f"Before NMS: {boxes_filt.shape[0]} boxes")
                nms_idx = torchvision.ops.nms(boxes_filt, scores.to(self.device), self.iou_threshold).cpu().numpy().tolist()
                boxes_filt = boxes_filt[nms_idx]
                pred_phrases = [pred_phrases[idx] for idx in nms_idx]
                # print(f"After NMS: {boxes_filt.shape[0]} boxes")
                caption = check_caption_spacy(caption, pred_phrases,caption_lemma)
                # print(f"Revise caption with number: {caption}")

                # sam
                image = image_tensor2cv2(image[None])
                self.predictor.set_image(image)
                transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

                try:
                    masks, _, _ = self.predictor.predict_torch(point_coords = None,point_labels = None,boxes = transformed_boxes,multimask_output = False,)
                except:
                    print(image.shape)
                    continue

                mask_img = torch.zeros(masks.shape[-2:])
                value = 0 
                for idx, mask in enumerate(masks):
                    mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1

                json_data = {
                        'caption': caption,
                        'mask':[{
                            'value': value,
                            'label': 'background'
                        }]
                    }
                for label, box in zip(pred_phrases, boxes_filt):
                    value += 1
                    name, logit = label.split('(')
                    logit = logit[:-1] # the last is ')'
                    json_data['mask'].append({
                        'value': value,
                        'label': name,
                        'logit': float(logit),
                        'box': box.cpu().numpy().tolist(),
                    })
                mask_img = image_tensor2pillow(mask_img)

                json_datas.append(json_data)
                mask_imgs.append(mask_img)
                cap_oris_new.append(cap_ori)
                keys_new.append(key)
                images_new.append(image_save)

            samples.append([keys_new,images_new,cap_oris_new,json_datas,mask_imgs])
        
        dst = wds.TarWriter(output)
        for sample in tqdm(samples,  desc=f"Writing {output}"):
            new_keys = [pre_name+name for name in sample[0]]
            for x,y,z,json,png in zip(new_keys,sample[1],sample[2],sample[3],sample[4]):
                dst.write({"__key__":x, "jpg":y, "txt":z,"json":json,"png":png})
            # dst.write({"__key__":new_keys, "jpg":sample[1], "txt":sample[2]})
            # dst.write({"__key__":str(new_keys), "jpg":str(sample[1]), "txt":str(sample[2])})
        dst.close()
        end = time()
        print(f"Finished - {end-start:.0f}s")


if __name__ == '__main__':
    device = "cuda"
    # origin_path = "/public_data/ma/aesthetics_tar_5"
    tar_begin = int(sys.argv[1])
    tar_end = int(sys.argv[2])
    origin_path = sys.argv[3]
    output_path = sys.argv[4]

    available_shards = list(range(tar_begin, tar_end))
    input_url = origin_path+"/{}.tar"
    input_shards = [input_url.format(str(shard).zfill(5)) for shard in available_shards]

    output_url = output_path+"/{}.tar"
    output_shards = [output_url.format(str(shard).zfill(5)) for shard in available_shards]

    data_pro = Data_pro(device)
    for input_shard, output_shard in zip(input_shards, output_shards):
        data_pro.shuffle_augment_wds(input=input_shard, output=output_shard)
