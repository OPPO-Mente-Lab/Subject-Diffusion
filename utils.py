import os
import json
from collections import OrderedDict
import torch
# from cn_clip.clip.configuration_bert import BertConfig
# from cn_clip.clip.modeling_bert import BertModel
from typing import Union, List
from PIL import Image

CONFIG_NAME = "RoBERTa-wwm-ext-large-chinese.json"
WEIGHT_NAME = "pytorch_model.bin"


def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images


def check_dir(save_directory):
    if os.path.isfile(save_directory):
        logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
        return

    os.makedirs(save_directory, exist_ok=True)


def save_images(images, save_directory, prompts, repeat=1):
    check_dir(save_directory)
    width, height = images[0].size
    assert len(images) == len(prompts) * repeat, "Input images has wrong number."
    for i in range(0, len(images), repeat):
        new_img = Image.new("RGB", (width*repeat, height))
        for j in range(repeat):
            new_img.paste(images[i+j], (width*j, 0))
        new_img.save(os.path.join(save_directory, "{}.png".format(prompts[int(i/repeat)])))

def save_config(bert_config, save_directory):
    check_dir(save_directory)
    # print(bert_config)
    dict_config = {
        "vocab_size": bert_config.vocab_size,
        "hidden_size": bert_config.hidden_size,
        "num_hidden_layers": bert_config.num_hidden_layers,
        "num_attention_heads": bert_config.num_attention_heads,
        "intermediate_size": bert_config.intermediate_size,
        "hidden_act": bert_config.hidden_act,
        "hidden_dropout_prob": bert_config.hidden_dropout_prob,
        "attention_probs_dropout_prob": bert_config.attention_probs_dropout_prob,
        "max_position_embeddings": bert_config.max_position_embeddings,
        "type_vocab_size": bert_config.type_vocab_size,
        "initializer_range": bert_config.initializer_range,
    }
    with open(os.path.join(save_directory, CONFIG_NAME), 'w', encoding='utf-8') as f:
        json.dump(dict_config, f, indent=4)


def save_model(model, save_directory):
    check_dir(save_directory)
    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(save_directory, WEIGHT_NAME))

def load_config(from_pretrained):
    with open(os.path.join(from_pretrained, CONFIG_NAME), 'r', encoding='utf-8') as f:
        config = json.load(f)

    bert_config = BertConfig(
        vocab_size_or_config_json_file=config["vocab_size"],
        hidden_size=config["hidden_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        intermediate_size=config["intermediate_size"],
        hidden_act=config["hidden_act"],
        hidden_dropout_prob=config["hidden_dropout_prob"],
        attention_probs_dropout_prob=config["attention_probs_dropout_prob"],
        max_position_embeddings=config["max_position_embeddings"],
        type_vocab_size=config["type_vocab_size"],
        initializer_range=config["initializer_range"],
        layer_norm_eps=1e-12,
    )
    return bert_config


def load_clip(from_pretrained, bert_config):
    # bert_config = load_config(from_pretrained)
    bert_model = BertModel(bert_config)
    with open(os.path.join(from_pretrained, WEIGHT_NAME), 'rb') as opened_file:
        # loading saved checkpoint
        checkpoint = torch.load(opened_file, map_location="cpu")
    if "state_dict" in checkpoint:
        sd = checkpoint["state_dict"]
    else:
        sd = checkpoint
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
    new_sd = OrderedDict()
    for key in sd:
        if key.startswith('bert'):
            new_sd[key[len('bert.'):]] = sd[key]
    if not new_sd:
        new_sd = sd
    print("load clip model ckpt from {}".format(os.path.join(from_pretrained, WEIGHT_NAME)))
    bert_model.load_state_dict(new_sd, strict=True)
    # bert_model = bert_model.to(device)
    return bert_model


def tokenize(tokenizer, texts: Union[str, List[str]], context_length: int = 64) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all baseline models use 24 as the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    all_tokens = []
    for text in texts:
        all_tokens.append([tokenizer.vocab['[CLS]']] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))[
                                                        :context_length - 2] + [tokenizer.vocab['[SEP]']])

    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        assert len(tokens) <= context_length
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

def tokenizer(tokenizer, texts: Union[str, List[str]], context_length: int = 64) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all baseline models use 24 as the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    all_tokens = []
    for text in texts:
        all_tokens.append([tokenizer.vocab['[CLS]']] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))[
                                                        :context_length - 2] + [tokenizer.vocab['[SEP]']])

    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        assert len(tokens) <= context_length
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result