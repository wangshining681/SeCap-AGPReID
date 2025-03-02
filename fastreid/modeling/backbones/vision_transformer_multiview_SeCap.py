""" Vision Transformer (ViT) in PyTorch
A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929
The official jax code is released and available at https://github.com/google-research/vision_transformer
Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.
Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert
Hacked together by / Copyright 2020 Ross Wightman
"""

import logging
import math
import pdb
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastreid.layers import DropPath, trunc_normal_, to_2tuple
from fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from .build import BACKBONE_REGISTRY
from functools import reduce
from operator import mul
import copy

logger = logging.getLogger(__name__)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):

    def __init__(
        self,
        embedding_dim: int,         # 输入channel
        num_heads: int,             # attention的head数
        downsample_rate: int = 1,   # 下采样
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        # qkv获取
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x, num_heads: int) :
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x):
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q, k, v) :
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        # B,N_heads,N_tokens,C_per_head
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B,N_heads,N_tokens,C_per_head
        # Scale
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        # Get output
        out = attn @ v
        # # B,N_tokens,C
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out

class OutAttenBlock(nn.Module):
    def __init__(self, dim, num_heads, 
                 mlp_ratio=4.0,
                 activation = nn.ReLU,   
                 attention_downsample_rate: int = 1, 
                 norm_layer=nn.LayerNorm):
        
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)
        self.norm3 = norm_layer(dim)

    def forward(self, q, k, v):
        attn_out = self.cross_attn(q, k, v)
        queries = q + attn_out
        queries = self.norm1(queries)
        queries = queries + self.attn(queries)
        queries = self.norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        return queries


class PromptRecapBlock(nn.Module):
    def __init__(self, 
                embedding_dim: int, 
                num_heads: int,
                activation = nn.ReLU,
                mlp_ratio = 4.0,
                method='attn'):
        super().__init__()
        self.cross_attn = CrossAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn = Attention(embedding_dim, num_heads)
        mlp_hidden_dim = int(embedding_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embedding_dim, hidden_features=mlp_hidden_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.method = method

        self.attn2 = Attention(embedding_dim, num_heads)

    def forward(self, queries, keys):
        if self.method == 'cat':
            queries = torch.cat((queries, keys), dim=1)
            attn_out = self.attn2(queries)
            queries = queries + attn_out
            queries = self.norm1(queries)
        elif self.method == 'add':
            queries = queries + keys
            queries = self.norm1(queries)
        elif self.method == 'attn':
            attn_out = self.cross_attn(q=queries, k=keys, v=keys)
            queries = queries + attn_out
            queries = self.norm1(queries)
        else:
            attn_out = self.cross_attn(q=queries, k=keys, v=keys)
            queries = queries + attn_out
            queries = self.norm1(queries)
        
        attn_out = self.attn(queries)
        queries = queries + attn_out
        queries = self.norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        if self.method == 'cat':
            k_len = keys.shape[1]
            # print(queries[:-k_len].shape)
            return queries[:, :-k_len]
        return queries

class PromptBlock(nn.Module):
    def __init__(self, 
                embedding_dim: int,         # 输入channel
                num_heads: int,             # attention的head数
                mlp_ratio: float = 4.0,        # MLP中间channel
                activation = nn.ReLU,      # 激活层
                attention_downsample_rate: int = 2,         # 下采样
                skip_first_layer_pe: bool = False,):
        super().__init__()
        self.cross_attn = CrossAttention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.self_attn = CrossAttention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)
        mlp_hidden_dim = int(embedding_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embedding_dim, hidden_features=mlp_hidden_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries, keys, query_pe, key_pe):
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        return queries, keys
    
class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,         # 输入channel
        num_heads: int,             # attention的head数
        mlp_ratio: float = 4.0,        # MLP中间channel
        activation = nn.ReLU,      # 激活层
        attention_downsample_rate: int = 2,         # 下采样
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = CrossAttention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = CrossAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        mlp_hidden_dim = int(embedding_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embedding_dim, hidden_features=mlp_hidden_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = CrossAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.skip_first_layer_pe = skip_first_layer_pe
        
    def forward(self, queries, keys, query_pe, key_pe):

        # queries：标记点编码相关(原始标记点编码经过一系列特征提取)
        # keys：原始图像编码相关(原始图像编码经过一系列特征提取)
        # query_pe：原始标记点编码
        # key_pe：原始图像位置编码
        # 第一轮本身queries==query_pe没比较再"残差"
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)
        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        return queries, keys

class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        # 层数
        depth: int,
        # 输入channel
        embedding_dim: int,
        # attention的head数
        num_heads: int,
        # MLP内部channel
        mlp_ratio: float,
        activation = nn.ReLU,
        attention_downsample_rate: int = 2,
        use_to_way = True,
        out_method = None,
    ) -> None:
        super().__init__()
        self.depth = depth      # 层数
        self.embedding_dim = embedding_dim          # 输入channel
        self.num_heads = num_heads                  # attention的head数
        self.mlp_ratio = mlp_ratio                      # MLP内部隐藏channel
        self.layers = nn.ModuleList()
        for i in range(depth):
            if use_to_way:
                self.layers.append(
                    TwoWayAttentionBlock(
                        embedding_dim=embedding_dim,    # 输入channel
                        num_heads=num_heads,            # attention的head数
                        mlp_ratio=mlp_ratio,                # MLP中间channel
                        activation=activation,          # 激活层
                        attention_downsample_rate=attention_downsample_rate,      # 下采样
                        skip_first_layer_pe=(i == 0),
                    )
                )
            else:
                self.layers.append(
                    PromptBlock(
                        embedding_dim=embedding_dim,    # 输入channel
                        num_heads=num_heads,            # attention的head数
                        mlp_ratio=mlp_ratio,                # MLP中间channel
                        activation=activation,          # 激活层
                        attention_downsample_rate=attention_downsample_rate,      # 下采样
                        skip_first_layer_pe=(i == 0),
                    )
                )
        self.out_method = out_method
        if out_method is None:
            self.final_attn_token_to_image = OutAttenBlock(
                embedding_dim, num_heads, mlp_ratio=self.mlp_ratio , attention_downsample_rate=attention_downsample_rate
            )
        elif out_method == 'Atten':
            self.final_attn_token_to_image = CrossAttention(
                embedding_dim, num_heads, downsample_rate=attention_downsample_rate
            )

        self.norm_final_attn = nn.LayerNorm(embedding_dim)
        self.out_token = nn.Parameter(torch.zeros(1, self.embedding_dim))
        trunc_normal_(self.out_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(
        self,
        image_embedding,
        image_pe,
        point_embedding,
        modality_flag
    ):
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, l, c = image_embedding.shape
        # 图像编码(image_encoder的输出)
        # BxHWxC=>B,N,C
        # image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        # 图像位置编码
        # BxHWxC=>B,N,C
        # image_pe = image_pe.flatten(2).permute(0, 2, 1)
        
        # 标记点编码
        # B,N,C
        queries = point_embedding
        queries_len = queries.shape[1]
        keys = image_embedding
        # -----TwoWayAttention-----
        
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )
        # -----TwoWayAttention-----

        q = queries + point_embedding
        # out_token = self.out_token.unsqueeze(0).expand(q.size(0), -1, -1)
        # q = torch.cat((out_token, q), dim=1)
        k = keys + image_pe
        if self.out_method == 'no_out':
            return queries, keys
        # -----Attention-----
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        # -----Attention-----
        queries = queries + attn_out
        # queries = attn_out
        queries = self.norm_final_attn(queries)
        return queries, keys

    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """

    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2)  # [64, 8, 768]
        return x


class VisionTransformer_multiview(nn.Module):
    """ Vision Transformer
        A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
            - https://arxiv.org/abs/2010.11929
        Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
            - https://arxiv.org/abs/2012.12877
        """

    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., camera=0, drop_path_rate=0., hybrid_backbone=None,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), sie_xishu=1.0, inner_sub=True, local_feat=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed_overlap(
                img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,
                embed_dim=embed_dim)

        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.view_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 2, embed_dim))
        self.cam_num = camera
        self.sie_xishu = sie_xishu
        self.local_feat = local_feat
        # Initialize SIE Embedding
        if camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)
        self.inner_sub = inner_sub

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'view_token'}

    def forward(self, x, camera_id=None):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        view_tokens = self.view_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, view_tokens, x), dim=1)

        if self.cam_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id]
        else:
            x = x + self.pos_embed

        x = self.pos_drop(x)
        if self.local_feat:
            for blk in self.blocks[:-1]:
                x = blk(x)
                # perform inner sub
                if self.inner_sub:
                    x[:, 0] = x[:, 0] - x[:, 1]
            return x

        else: 
            for blk in self.blocks:
                x = blk(x)
                # perform inner sub
                if self.inner_sub:
                    x[:, 0] = x[:, 0] - x[:, 1]
            
            x = self.norm(x)
            return x[:, 0].reshape(x.shape[0], -1, 1, 1), x[:, 1].reshape(x.shape[0], -1, 1, 1)
            
    def load_param(self, pretrain_path):
        try:
            state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
            logger.info(f"Loading pretrained model from {pretrain_path}")

            if 'model' in state_dict:
                state_dict = state_dict.pop('model')
            if 'state_dict' in state_dict:
                state_dict = state_dict.pop('state_dict')
            for k, v in state_dict.items():
                if 'head' in k or 'dist' in k:
                    continue
                if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                    # For old models that I trained prior to conv based patchification
                    O, I, H, W = self.patch_embed.proj.weight.shape
                    v = v.reshape(O, -1, H, W)
                elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                    # To resize pos embedding when using model at different size from pretrained weights
                    if 'distilled' in pretrain_path:
                        logger.info("distill need to choose right cls token in the pth.")
                        v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                    v = resize_pos_embed(v, self.pos_embed.data, self.patch_embed.num_y, self.patch_embed.num_x, 2)
                state_dict[k] = v
        except FileNotFoundError as e:
            logger.info(f'{pretrain_path} is not found! Please check this path.')
            raise e
        except KeyError as e:
            logger.info("State dict keys error! Please check the state dict.")
            raise e

        incompatible = self.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logger.info(
                get_missing_parameters_message(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )

def resize_pos_embed(posemb, posemb_new, hight, width, cls_token_num):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :cls_token_num], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    logger.info('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape,
                                                                                                      posemb_new.shape,
                                                                                                      hight,
                                                                                                      width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb

class Vision_Transformer_SeCap(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # fmt: off
        input_size = cfg.INPUT.SIZE_TRAIN
        pretrain = cfg.MODEL.BACKBONE.PRETRAIN
        pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
        depth = cfg.MODEL.BACKBONE.DEPTH
        sie_xishu = cfg.MODEL.BACKBONE.SIE_COE
        stride_size = cfg.MODEL.BACKBONE.STRIDE_SIZE
        drop_ratio = cfg.MODEL.BACKBONE.DROP_RATIO
        drop_path_ratio = cfg.MODEL.BACKBONE.DROP_PATH_RATIO
        attn_drop_rate = cfg.MODEL.BACKBONE.ATT_DROP_RATE
        inner_sub = cfg.MODEL.BACKBONE.INNER_SUB
        self.in_planes = 768
        self.prompt_len = cfg.MODEL.BACKBONE.PROMPT_LEN
        self.prompt_trans_depth = cfg.MODEL.BACKBONE.PROMPT_DEPTH
        # self.use_prompt = cfg.MODEL.BACKBONE.USE_PROMPT
        pretrain = cfg.MODEL.BACKBONE.PRETRAIN
        pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
        self.use_prm = cfg.MODEL.BACKBONE.USE_PRM
        # fmt: on
        
        # VDT init
        num_depth = {'small': 8, 'base': 12,}[depth]
        num_heads = {'small': 8, 'base': 12,}[depth]
        mlp_ratio = {'small': 3., 'base': 4,}[depth]
        qkv_bias = {'small': False, 'base': True}[depth]
        qk_scale = {'small': 768 ** -0.5, 'base': None,}[depth]
        self.base = VisionTransformer_multiview(
            img_size=input_size, sie_xishu=sie_xishu, stride_size=stride_size,
            depth=num_depth,
            num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_path_rate=drop_path_ratio, drop_rate=drop_ratio,
            attn_drop_rate=attn_drop_rate, inner_sub=inner_sub, local_feat=True
        )

        if pretrain:
            self.base.load_param(pretrain_path)

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        # PRM init
        self.__init_query__(self.prompt_len, self.in_planes)
        self.prm = PromptRecapBlock(embedding_dim=self.in_planes, num_heads=num_heads)
        
        # LFRM init
        self.lfrm = TwoWayTransformer(
            # 层数
            depth=self.prompt_trans_depth,
            # 输入channel
            embedding_dim=self.in_planes,
            # MLP内部channel
            mlp_ratio=mlp_ratio,
            # attention的head数
            num_heads=num_heads,
            use_to_way=True,
        )
        self.image_pe = nn.Parameter(torch.zeros(1, self.base.num_patches, self.in_planes))
        trunc_normal_(self.image_pe, std=.02)
        self.out_token = nn.Parameter(torch.zeros(1, self.in_planes))
        trunc_normal_(self.out_token, std=.02)

    def __init_query__(self, prompt_len, prompt_dim, prompt_scale=20.0, prompt_shift=0.0):
        self.prompt = nn.Parameter(torch.zeros(1, prompt_len, prompt_dim)) # type: ignore
        trunc_normal_(self.prompt, std=.02)

    def forward(self, x, camera_id=None):
        B = x.shape[0]
        # VDT
        local_features = self.base(x, camera_id=camera_id)
        local_feat = self.b1(local_features)
        global_features = local_feat[:, 0:1]
        view_features = local_feat[:, 1:2]
        local_feat = local_features[:, 2:]
        inv_features = global_features - view_features

        # PRM
        query_feat = torch.repeat_interleave(self.prompt, B, dim=0)
        if self.use_prm:
            Re_Prompt = self.prm(query_feat, inv_features)
        else:
            Re_Prompt = query_feat

        # LFRM
        out_token = self.out_token.unsqueeze(0).expand(B, -1, -1)
        prompt = torch.cat((out_token, Re_Prompt), dim=1)
        pos_src = torch.repeat_interleave(self.image_pe, B, dim=0)
        prompts, img_feature = self.lfrm(local_feat, pos_src, prompt, camera_id)
        out_feat = prompts[:, 0, :]

        return  global_features.reshape(x.shape[0], -1, 1, 1), out_feat.reshape(x.shape[0], -1, 1, 1), view_features.reshape(x.shape[0], -1, 1, 1),


@BACKBONE_REGISTRY.register()
def build_multiview_vit_backbone_SeCap(cfg):
    """
    Create a Vision Transformer instance from config.
    Returns:
        SwinTransformer: a :class:`SwinTransformer` instance.
    """
    # fmt: off

    # fmt: on

    model = Vision_Transformer_SeCap(cfg)
    return model
