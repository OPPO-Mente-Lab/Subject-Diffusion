import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x


class FourierEmbedder():
    def __init__(self, num_freqs=64, temperature=100):

        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** (torch.arange(num_freqs) / num_freqs)

    @ torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        "x: arbitrary shape of tensor. dim: cat dim"
        out = []
        for freq in self.freq_bands:
            out.append(torch.sin(freq*x))
            out.append(torch.cos(freq*x))
        return torch.cat(out, cat_dim)


class GroundingNet(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 fourier_freqs=8,
                 num_token=256,
                 use_bbox=True
                 ):
        super(GroundingNet, self).__init__()
        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs*2*4
        self.linears_image = MLP(
            in_dim=input_dim + self.position_dim, out_dim=output_dim, hidden_dim=hidden_dim, use_residual=False)
        self.null_image_feature = torch.nn.Parameter(
            torch.zeros([1, 1, num_token, input_dim]))
        self.null_position_feature = torch.nn.Parameter(
            torch.zeros([1, 1, num_token, self.position_dim]))
        self.use_bbox = use_bbox

    def forward(self, image_embeddings, image_token_idx_mask, bboxes):
        bsz, num_of_objects, _, dim = image_embeddings.size()
        image_embeddings = image_embeddings*image_token_idx_mask + \
            (~image_token_idx_mask)*self.null_image_feature
        xyxy_embedding = self.fourier_embedder(
            bboxes).unsqueeze(-2)  # B*N*4 --> B*N*C
        if not self.use_bbox:
            image_token_idx_mask = image_token_idx_mask.sum(
                1, keepdim=True) > 1
        xyxy_embedding = xyxy_embedding*image_token_idx_mask + \
            (~image_token_idx_mask)*self.null_position_feature
        xyxy_embedding = xyxy_embedding.reshape(bsz, -1, self.position_dim)
        image_embeddings = image_embeddings.reshape(bsz, -1, dim)
        objs_image = self.linears_image(
            torch.cat([image_embeddings, xyxy_embedding], dim=-1))
        return objs_image
