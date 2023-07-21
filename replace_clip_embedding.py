import types
import torch
from typing import Optional


def replace_clip_embeddings(clip_model, image_infos):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        if image_infos["image_token_mask"] is not None:

            shape_mask = inputs_embeds[image_infos["image_token_mask"]].shape[0]
            shape_embedding = image_infos["image_embedding"].shape[0]
            assert shape_mask == shape_embedding or (
                shape_mask == 4 and shape_embedding == 1)
            inputs_embeds[image_infos["image_token_mask"]
                          ] = image_infos["image_embedding"]

        embeddings = inputs_embeds + position_embeddings
        return embeddings
    clip_model.text_model.embeddings.old_forward = clip_model.text_model.embeddings.forward
    clip_model.text_model.embeddings.forward = types.MethodType(
        forward, clip_model.text_model.embeddings
    )
