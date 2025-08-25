import math
import logging
import pdb
from functools import partial, reduce
from collections import OrderedDict
from copy import deepcopy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens, token2feature, feature2token
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock, candidate_elimination_adapter


from ..layers.attn_adapt_blocks import CEABlock,CEABlock_Enhancement, CEABlock_Reconstruction   ##BAT 
from ..layers.dualstream_attn_blocks import DSBlock ## Dual Stream without adapter


from lib.models.layers.attn import Attention
from lib.models.layers.adapter import Bi_direct_adapter

from lib.models.bat.uncertainty_fusion import CrossModal_ST_Fusion_with_uncertainty,BasicBlock,TemplateRouter

_logger = logging.getLogger(__name__)


class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', ce_loc=None, ce_keep_ratio=None, search_size=None, template_size=None,
                 new_patch_size=None, adapter_type=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
            new_patch_size: backbone stride
        """
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        #self.patch_embed_adapter = embed_layer(
        #    img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim)) # it's redundant
        self.pos_drop = nn.Dropout(p=drop_rate)

        
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search=new_P_H * new_P_W
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template=new_P_H * new_P_W
        """add here, no need use backbone.finetune_track """     #
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))

        depth = 12
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:  #ce_loc [3,6,9]
                ce_keep_ratio_i = ce_keep_ratio[ce_index]  #[1,1,1]
                ce_index += 1
            if i ==9 or i ==10 or i == 11:
                blocks.append(
                CEABlock_Enhancement(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                        keep_ratio_search=ce_keep_ratio_i)
                )
              
            elif i<20:
                blocks.append(
                CEABlock(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                        keep_ratio_search=ce_keep_ratio_i)
                )
            else:
                blocks.append(
                    DSBlock(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                        keep_ratio_search=ce_keep_ratio_i)
                )
        

        self.blocks = nn.Sequential(*blocks)       
        self.norm = norm_layer(embed_dim)
        self.init_weights(weight_init)
        self.templaterouter = TemplateRouter(64,embed_dim)

    
    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False,dynamic_template=None,Test=None,template_masks=None):

        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        if Test is None:
            z1,z2 = z[0],z[1]
            x_rgb = x[:, :3, :, :]
            x_dte = x[:, 3:, :, :]

            z1_rgb = z1[:, :3, :, :]
            z1_dte = z1[:, 3:, :, :]
            z2_rgb = z2[:, :3, :, :]
            z2_dte = z2[:, 3:, :, :]
            z_list,zi_list = [],[]
            for i in range(B):
                z_list.append(torch.cat([z1_rgb[i].unsqueeze(0),z2_rgb[i].unsqueeze(0)],dim=0))
                zi_list.append(torch.cat([z1_dte[i].unsqueeze(0),z2_dte[i].unsqueeze(0)],dim=0))
            z_rgb = torch.cat(z_list,dim=0)
            z_dte = torch.cat(zi_list,dim=0)

            x_list,xi_list = [],[]
            for i in range(B):
                x_list.append(torch.cat([x_rgb[i].unsqueeze(0),x_rgb[i].unsqueeze(0)],dim=0))
                xi_list.append(torch.cat([x_dte[i].unsqueeze(0),x_dte[i].unsqueeze(0)],dim=0))
            x_rgb = torch.cat(x_list,dim=0)
            x_dte = torch.cat(xi_list,dim=0)
        else:
        # rgb_img
            x_rgb = x[:, :3, :, :]
            z_rgb = z[:, :3, :, :]
            # depth thermal event images
            x_dte = x[:, 3:, :, :]
            z_dte = z[:, 3:, :, :]
        # overwrite x & z
        x, z = x_rgb, z_rgb
        xi, zi = x_dte, z_dte
        self.test = Test
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        #print("input x",x.size())

        z = self.patch_embed(z)
        x = self.patch_embed(x)

        #print("after patch_embed x",x.size())

        xi = self.patch_embed(xi)
        zi = self.patch_embed(zi)

        if dynamic_template == None:
            pass
        else:
            self.dynamic_template = dynamic_template

###################################################################===========
        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        zi += self.pos_embed_z
        xi += self.pos_embed_x



        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed
            xi += self.search_segment_pos_embed      #//////////////////////////////////////////////////////////////////
            zi += self.template_segment_pos_embed
        #print(x.shape) #[Batch size, 256, 768]
        #z [bs,64,768]
        x = combine_tokens(z, x, mode=self.cat_mode)  ##[Batch size, 320, 768]
        #print("after cat",x.shape)

        xi = combine_tokens(zi, xi, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)
            xi = torch.cat([cls_tokens, xi], dim=1)

        x = self.pos_drop(x)
        xi = self.pos_drop(xi)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z, dtype=torch.int64).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)

        global_index_s = torch.linspace(0, lens_x - 1, lens_x, dtype=torch.int64).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)

        global_index_ti = torch.linspace(0, lens_z - 1, lens_z, dtype=torch.int64).to(x.device)
        global_index_ti = global_index_ti.repeat(B, 1)

        global_index_si = torch.linspace(0, lens_x - 1, lens_x, dtype=torch.int64).to(x.device)
        global_index_si = global_index_si.repeat(B, 1)
        if Test is not None:
            B,L,C = x.shape
            z,zi = x[:,:64],xi[:,:64]
            x_new,xi_new = x[:,64:],xi[:,64:]
            if Test:
                z_new = z.clone()
                zi_new = zi.clone()
                # for i in range(1,B):
                #     z_new[0] = z[i].unsqueeze(0)
                #     zi_new[0] = zi[i].unsqueeze(0)
                #     z_new[0] = z[1].unsqueeze(0)
                #     zi_new[0] = zi[1].unsqueeze(0)
                """
                I D I C/ D I I I
                """
                if B == 4:
                    zi_new[2] = zi[0].unsqueeze(0)
                    z_new[2] = z[0].unsqueeze(0)
                    z_new[3] = z[0].unsqueeze(0)
                    zi_new[3] = zi[0].unsqueeze(0)
                    z_new[0] = z[1].unsqueeze(0)
                    zi_new[0] = zi[1].unsqueeze(0)
                    z_new[1] = z[0].unsqueeze(0)
                    zi_new[1] = zi[0].unsqueeze(0)
                else:
                    z_new[1] = z[0].unsqueeze(0)
                    zi_new[1] = zi[0].unsqueeze(0)
                    z_new[0] = z[1].unsqueeze(0)
                    zi_new[0] = zi[1].unsqueeze(0)
                x = torch.cat([z,z_new,x_new],dim=1)
                xi = torch.cat([zi,zi_new,xi_new],dim=1)



        removed_indexes_s = []
        removed_indexes_si = []
        #用于统计两个模态间ce的差值
        # diff_sum = 0
        x_list = []
        test_tokens = True
        removed_flag = False
  
        paramters = []



        for i, blk in enumerate(self.blocks[:12]):
            if i ==9 or i ==10 or i == 11:
                x, global_index_t, global_index_s, removed_index_s, attn, \
                xi, global_index_ti, global_index_si, removed_index_si, attn_i,alpha_m1,alpha_m2,[u_m,u,ui] = \
                    blk(x, xi, global_index_t, global_index_ti, global_index_s, global_index_si, mask_x, ce_template_mask,
                        ce_keep_rate,self.test,dynamic_template,template_masks=template_masks)
                if self.ce_loc is not None and i in self.ce_loc:
                    removed_indexes_s.append(removed_index_s)
                    removed_indexes_si.append(removed_index_si)
                paramters.append(alpha_m1)
                paramters.append(alpha_m2)

            else:
                x, global_index_t, global_index_s, removed_index_s, attn, \
                xi, global_index_ti, global_index_si, removed_index_si, attn_i = \
                    blk(x, xi, global_index_t, global_index_ti, global_index_s, global_index_si, mask_x, ce_template_mask,
                        ce_keep_rate,dynamic_template)
                if self.ce_loc is not None and i in self.ce_loc:
                    removed_indexes_s.append(removed_index_s)
                    removed_indexes_si.append(removed_index_si)
            if i == 9:
                self.um = [u_m,u,ui]
            # if i == 6:
            #     x,xi = self.adap_selfattn_uncerttainty1(x,xi)
            # if i == 8:
            #     x,xi = self.adap_selfattn_uncerttainty2(x,xi)
            # if i == 10:
            #     x,xi = self.adap_selfattn_uncerttainty3(x,xi)



        if Test:
            x,xi = x[:,64:],xi[:,64:]


        # x = x+x_
        # xi = xi+xi_
        x = self.norm(x)
        xi = self.norm(xi)


        # x,xi,T,[u_m,u,ui],alpha_m =  self.uncertainty_fusion(x,xi)
        # paramters.append(alpha_m)       

        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]
        lens_xi_new = global_index_si.shape[1]
        lens_zi_new = global_index_ti.shape[1]

        z = x[:, :lens_z_new]
        x = x[:, lens_z_new:]
        zi = xi[:, :lens_zi_new]
        xi = xi[:, lens_zi_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)
        
        if removed_indexes_si and removed_indexes_si[0] is not None:
            removed_indexes_cat_i = torch.cat(removed_indexes_si, dim=1)

            pruned_lens_xi = lens_x - lens_xi_new                                ########################
            pad_xi = torch.zeros([B, pruned_lens_xi, xi.shape[2]], device=xi.device)
            xi = torch.cat([xi, pad_xi], dim=1)
            index_all = torch.cat([global_index_si, removed_indexes_cat_i], dim=1)
            # recover original token order
            C = xi.shape[-1]
            # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
            xi = torch.zeros_like(xi).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=xi)
        
        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)
        xi = recover_tokens(xi, lens_zi_new, lens_x, mode=self.cat_mode)


        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([z, x], dim=1)
        xi = torch.cat([zi, xi], dim=1)
        #x = torch.cat([x, xi], dim=0)
        #print("===========final out: ",x.size())
        # x = self.adap_fusion(x,xi)
        x = x + xi

        T = x[:,:64]
        scores_list, relative_score_list = [],[]
        for j in range(0,B,2):
            t1,t2 = T[j].unsqueeze(0),T[j+1].unsqueeze(0)
            scores, relative_score = self.templaterouter(t1,t2)
            scores_list.append(scores)
            relative_score_list.append(relative_score)
        
        scores,relative_score = torch.cat(scores_list,dim=0),torch.cat(relative_score_list,dim=0)

        # x_fusion = self.adap_crossstage(x,x10)
        # x = x+x_fusion
        # x = torch.cat([x, xi], dim=1)
        #x = torch.cat([x, xi], dim=2)
        #x = self.adap_headcat(x)
        # x = self.adap_conv(x)

        #print("-------",x.shape)
        #print("attn",attn.size())
        aux_dict = {
            "attn": attn,
            # "u":u,
            # "ui":ui,
            "u_m":[u_m,u,ui],
            "weights":paramters,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
            "relative_score":relative_score,
            "score":scores
        }
        # print('diff_sum',diff_sum)
        return x, aux_dict,dynamic_template

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False,dynamic_template=None,Test=None,template_masks=None):
        dynamic_template_ = dynamic_template

  
        x, aux_dict,dynamic_template = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,dynamic_template=dynamic_template_,Test=Test,template_masks=template_masks)
        # dynamic_template_save = dynamic_template.detach()
        dynamic_template_save = None
        return x, aux_dict,dynamic_template_save


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained OSTrack from: ' + pretrained)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")

    return model


def vit_base_patch16_224_ce_adapter(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce_adapter(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
