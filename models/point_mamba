from typing import Union, Optional
import math
import random
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from mamba_ssm.modules.mamba_simple import Mamba
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from knn_cuda import KNN
from .block import Block
from .build import MODELS

class Encoder(nn.Module):  ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        # import ipdb; ipdb.set_trace()
        # idx = knn_query(xyz, center, self.group_size)  # B G M
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block

class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out_in_block: int = 0.,
            drop_path: int = 0.1,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids  # + pos
        residual = None
        hidden_states = hidden_states + pos
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            hidden_states = self.drop_out_in_block(hidden_states)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states

@MODELS.register_module()
class PointMamba(nn.Module):
    def __init__(self, config, **kwargs):
        super(PointMamba, self).__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.cls_dim = config.cls_dim

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.use_cls_token = False if not hasattr(self.config, "use_cls_token") else self.config.use_cls_token
        self.drop_path = 0. if not hasattr(self.config, "drop_path") else self.config.drop_path
        self.rms_norm = False if not hasattr(self.config, "rms_norm") else self.config.rms_norm
        self.drop_out_in_block = 0. if not hasattr(self.config, "drop_out_in_block") else self.config.drop_out_in_block

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.cls_pos, std=.02)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.blocks = MixerModel(d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 rms_norm=self.rms_norm,
                                 drop_out_in_block=self.drop_out_in_block,
                                 drop_path=self.drop_path)

        self.norm = nn.LayerNorm(self.trans_dim)

        self.HEAD_CHANEL = 1
        if self.use_cls_token:
            self.HEAD_CHANEL += 1

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * self.HEAD_CHANEL, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func()

        self.drop_out = nn.Dropout(config.drop_out) if "drop_out" in config else nn.Dropout(0)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Mamba')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Mamba'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Mamba')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Mamba'
                )

            print_log(f'[Mamba] Successful Loading the ckpt from {bert_ckpt_path}', logger='Mamba')
        else:
            print_log('Training from scratch!!!', logger='Mamba')
            self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        pos = self.pos_embed(center)

        # reordering strategy
        center_x = center[:, :, 0].argsort(dim=-1)[:, :, None]
        center_y = center[:, :, 1].argsort(dim=-1)[:, :, None]
        center_z = center[:, :, 2].argsort(dim=-1)[:, :, None]
        group_input_tokens_x = group_input_tokens.gather(dim=1, index=torch.tile(center_x, (
            1, 1, group_input_tokens.shape[-1])))
        group_input_tokens_y = group_input_tokens.gather(dim=1, index=torch.tile(center_y, (
            1, 1, group_input_tokens.shape[-1])))
        group_input_tokens_z = group_input_tokens.gather(dim=1, index=torch.tile(center_z, (
            1, 1, group_input_tokens.shape[-1])))
        pos_x = pos.gather(dim=1, index=torch.tile(center_x, (1, 1, pos.shape[-1])))
        pos_y = pos.gather(dim=1, index=torch.tile(center_y, (1, 1, pos.shape[-1])))
        pos_z = pos.gather(dim=1, index=torch.tile(center_z, (1, 1, pos.shape[-1])))
        group_input_tokens = torch.cat([group_input_tokens_x, group_input_tokens_y, group_input_tokens_z],
                                       dim=1)
        pos = torch.cat([pos_x, pos_y, pos_z], dim=1)

        x = group_input_tokens
        # transformer
        x = self.drop_out(x)
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = x[:, :].mean(1)
        ret = self.cls_head_finetune(concat_f)
        return ret

class MaskMamba(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.num_heads = config.transformer_config.num_heads
        print_log(f'[args] {config.transformer_config}', logger='Mamba')
        
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.mask_type = config.transformer_config.mask_type
        self.semantic_threshold = getattr(config.transformer_config, 'semantic_threshold', 0.6)
        
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )
        
        self.blocks = MixerModel(d_model=self.trans_dim,
                               n_layer=self.depth,
                               rms_norm=self.config.rms_norm)
        
        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            -----------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        
        self.num_mask = int(self.mask_ratio * G)
        
        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
            
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)
        return overall_mask.to(center.device)  # B G
    
    def Second_remove_redundant_tokens(self, group_input_tokens, threshold=0.6):
        B, G, C = group_input_tokens.shape
        tokens_norm = F.normalize(group_input_tokens, dim=-1)
        similarity_matrix = torch.bmm(tokens_norm, tokens_norm.transpose(1, 2)).clamp(0, 1)
        redundancy_score = similarity_matrix.sum(dim=-1)  # B x G
        k = int(threshold * G) 
        if k == 0: 
            return torch.zeros((B, G), dtype=torch.bool, device=group_input_tokens.device)
        thresholds = torch.topk(redundancy_score, k=k, dim=-1, largest=False).values[:, -1]
        bool_masked_pos = redundancy_score > thresholds.unsqueeze(-1)
        
        return bool_masked_pos
    
    def forward(self, neighborhood, center, noaug=False):
        B, G, _, _ = neighborhood.shape
        
        group_input_tokens = self.encoder(neighborhood)  # B G C
        
        if self.mask_type == 'rand_semantic' and not noaug:
            bool_masked_semantic = self.Second_remove_redundant_tokens(group_input_tokens, threshold=self.semantic_threshold)  # B G, True 表示语义掩码
            
            remaining_centers_list = []
            remaining_indices_list = []
            
            for b in range(B):
                unmasked_indices = torch.where(~bool_masked_semantic[b])[0]
                remaining_centers_list.append(center[b, unmasked_indices])
                remaining_indices_list.append(unmasked_indices)
            
            bool_masked_rand = torch.zeros_like(bool_masked_semantic)
            
            for b in range(B):
                remaining_centers = remaining_centers_list[b]
                remaining_indices = remaining_indices_list[b]
                
                if len(remaining_centers) > 0:
                    temp_center = remaining_centers.unsqueeze(0)  
                    temp_mask = self._mask_center_rand(temp_center, noaug=noaug) 
                    bool_masked_rand[b, remaining_indices] = temp_mask[0]
            bool_masked_pos = bool_masked_semantic | bool_masked_rand  # B G
            
        elif self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)
        else:
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)
        
        batch_size, num_group, C = group_input_tokens.shape
        
        original_mask = bool_masked_pos.clone()
        for b in range(batch_size):
            if torch.all(bool_masked_pos[b]):
                random_idx = torch.randint(0, num_group, (1,), device=bool_masked_pos.device)
                bool_masked_pos[b, random_idx] = False
                print(f"[Warning] Batch {b} has all tokens masked. Randomly unmasking one token.")
        unmasked_counts = torch.sum(~bool_masked_pos, dim=1)  # [B]
        min_unmasked = unmasked_counts.min().item()
        x_vis_list = []
        vis_center_list = []
        
        for b in range(batch_size):
            unmasked_indices = torch.where(~bool_masked_pos[b])[0]
            if len(unmasked_indices) > min_unmasked:
                perm = torch.randperm(len(unmasked_indices), device=unmasked_indices.device)
                unmasked_indices = unmasked_indices[perm[:min_unmasked]]
            b_vis_tokens = group_input_tokens[b, unmasked_indices]
            b_vis_centers = center[b, unmasked_indices]
            if len(b_vis_tokens) < min_unmasked:
                num_to_add = min_unmasked - len(b_vis_tokens)
                if len(b_vis_tokens) > 0:
                    indices = torch.randint(0, len(b_vis_tokens), (num_to_add,), device=b_vis_tokens.device)
                    b_vis_tokens = torch.cat([b_vis_tokens, b_vis_tokens[indices]], dim=0)
                    b_vis_centers = torch.cat([b_vis_centers, b_vis_centers[indices]], dim=0)
                else:
                    b_vis_tokens = torch.randn(min_unmasked, C, device=group_input_tokens.device)
                    b_vis_centers = torch.randn(min_unmasked, 3, device=center.device)
            assert b_vis_tokens.shape[0] == min_unmasked, f"Expected {min_unmasked} tokens, got {b_vis_tokens.shape[0]}"
            
            x_vis_list.append(b_vis_tokens)
            vis_center_list.append(b_vis_centers)
        x_vis = torch.stack(x_vis_list, dim=0)  # [B, min_unmasked, C]
        vis_centers = torch.stack(vis_center_list, dim=0)  # [B, min_unmasked, 3]
        
        pos = self.pos_embed(vis_centers)
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)
        return x_vis, original_mask

class MambaDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, norm_layer=nn.LayerNorm, config=None): 
        super().__init__()
        if hasattr(config, "use_external_dwconv_at_last"):
            self.use_external_dwconv_at_last = config.use_external_dwconv_at_last
        else:
            self.use_external_dwconv_at_last = False
        self.blocks = MixerModel(d_model=embed_dim,
                                 n_layer=depth,
                                 rms_norm=config.rms_norm,
                                 drop_path=config.drop_path)
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        x = self.blocks(x, pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x


@MODELS.register_module()
class Point_MAE_Mamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MAE] ', logger='Point_MAE')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskMamba(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        
        self.decoder_depth = config.transformer_config.decoder_depth
        self.MAE_decoder = MambaDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            config=config,
        )
        
        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                 logger='Point_MAE')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        
        # prediction head
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )
        
        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        
        # loss
        self.build_loss_func(self.loss)
        
    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
    
    def forward(self, pts, vis=False, **kwargs):
        neighborhood, center = self.group_divider(pts)
        x_vis, mask = self.MAE_encoder(neighborhood, center)
        
        B, N_vis, C = x_vis.shape 
        for b in range(B):
            if not torch.any(mask[b]): 
                random_idx = torch.randint(0, self.num_group, (1,), device=mask.device)
                mask[b, random_idx] = True
                print(f"[Warning] Batch {b} has no masked tokens. Randomly masking one token.")
        
        masked_counts = torch.sum(mask, dim=1)  # [B]
        min_masked = masked_counts.min().item()
        mask_center_list = []
        mask_neighborhood_list = []
        
        for b in range(B):
            masked_indices = torch.where(mask[b])[0]
            
            if len(masked_indices) > min_masked:
                perm = torch.randperm(len(masked_indices), device=masked_indices.device)
                masked_indices = masked_indices[perm[:min_masked]]
            
            b_mask_centers = center[b, masked_indices]
            b_mask_neighborhood = neighborhood[b, masked_indices]
            if len(b_mask_centers) < min_masked:
                num_to_add = min_masked - len(b_mask_centers)
                if len(b_mask_centers) > 0:
                    indices = torch.randint(0, len(b_mask_centers), (num_to_add,), device=b_mask_centers.device)
                    b_mask_centers = torch.cat([b_mask_centers, b_mask_centers[indices]], dim=0)
                    b_mask_neighborhood = torch.cat([b_mask_neighborhood, b_mask_neighborhood[indices]], dim=0)
                else:
                    b_mask_centers = torch.randn(min_masked, 3, device=center.device)
                    b_mask_neighborhood = torch.randn(min_masked, self.group_size, 3, device=neighborhood.device)
            
            assert b_mask_centers.shape[0] == min_masked, f"Expected {min_masked} tokens, got {b_mask_centers.shape[0]}"
            
            mask_center_list.append(b_mask_centers)
            mask_neighborhood_list.append(b_mask_neighborhood)
        mask_centers = torch.stack(mask_center_list, dim=0)  
        mask_neighborhood = torch.stack(mask_neighborhood_list, dim=0) 
        vis_centers = torch.zeros(B, N_vis, 3, device=center.device)
        for b in range(B):
            unmasked_indices = torch.where(~mask[b])[0]
            if len(unmasked_indices) > N_vis:
                perm = torch.randperm(len(unmasked_indices), device=unmasked_indices.device)
                unmasked_indices = unmasked_indices[perm[:N_vis]]
            elif len(unmasked_indices) < N_vis:
                indices = torch.randint(0, len(unmasked_indices), (N_vis - len(unmasked_indices),), device=unmasked_indices.device)
                additional_indices = unmasked_indices[indices]
                unmasked_indices = torch.cat([unmasked_indices, additional_indices])
            
            vis_centers[b] = center[b, unmasked_indices[:N_vis]]
        
        pos_emd_vis = self.decoder_pos_embed(vis_centers) 
        pos_emd_mask = self.decoder_pos_embed(mask_centers) 
        
        mask_token = self.mask_token.expand(B, min_masked, -1) 
        
        x_full = torch.cat([x_vis, mask_token], dim=1)  
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)  
        x_rec = self.MAE_decoder(x_full, pos_full, min_masked)  
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)) 
        rebuild_points = rebuild_points.transpose(1, 2) 
        rebuild_points = rebuild_points.reshape(B * min_masked, self.group_size, 3) 
        gt_points = mask_neighborhood.reshape(B * min_masked, self.group_size, 3)
        
        loss1 = self.loss_func(rebuild_points, gt_points)
        
        if vis:
            vis_points_list = []
            rebuild_points_list = []
            visible_center_list = []
            masked_center_list = []
            
            for b in range(B):
                unmasked_indices = torch.where(~mask[b])[0]
                if len(unmasked_indices) > 0:
                    if len(unmasked_indices) > N_vis:
                        unmasked_indices = unmasked_indices[:N_vis]
                    elif len(unmasked_indices) < N_vis:
                        indices = torch.randint(0, len(unmasked_indices), (N_vis - len(unmasked_indices),), device=unmasked_indices.device)
                        additional_indices = unmasked_indices[indices]
                        unmasked_indices = torch.cat([unmasked_indices, additional_indices])
                    
                    b_vis_points = neighborhood[b, unmasked_indices[:N_vis]]
                    b_vis_centers = center[b, unmasked_indices[:N_vis]]
                    full_vis = b_vis_points + b_vis_centers.unsqueeze(1)
                    vis_points_list.append(full_vis)
                    visible_center_list.append(b_vis_centers)
                b_rebuild = rebuild_points[b * min_masked:(b + 1) * min_masked]
                b_mask_centers = mask_centers[b]
                full_rebuild = b_rebuild + b_mask_centers.unsqueeze(1)
                rebuild_points_list.append(full_rebuild)
                masked_center_list.append(b_mask_centers)
            if vis_points_list: 
                all_vis_points = torch.cat(vis_points_list, dim=0)
                ret2 = all_vis_points.reshape(-1, 3).unsqueeze(0)
            else:
                ret2 = torch.zeros(1, 1, 3, device=center.device)
            
            if rebuild_points_list: 
                all_rebuild_points = torch.cat(rebuild_points_list, dim=0)
                
                if vis_points_list:  
                    full = torch.cat([all_vis_points, all_rebuild_points], dim=0)
                else:
                    full = all_rebuild_points
                
                ret1 = full.reshape(-1, 3).unsqueeze(0)
            else:
                ret1 = torch.zeros(1, 1, 3, device=center.device)
            
            all_centers = []
            if visible_center_list:
                all_centers.append(torch.cat(visible_center_list, dim=0))
            if masked_center_list:
                all_centers.append(torch.cat(masked_center_list, dim=0))
            
            if all_centers:
                full_center = torch.cat(all_centers, dim=0)
            else:
                full_center = torch.zeros(1, 3, device=center.device)
            
            return ret1, ret2, full_center
        else:
            return loss1
