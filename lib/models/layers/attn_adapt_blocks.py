import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

from lib.models.layers.attn import Attention
from lib.models.layers.adapter import Bi_direct_adapter
from lib.models.bat.cross_layer import CrossModal_Templates_Update,CrossModal_ST_Fusion,CrossAttention,CrossModal_ST_Fusion_Temporal
from lib.models.bat.uncertainty_fusion import CrossModal_ST_Fusion_with_uncertainty,BasicBlock
class FeatureReconstructionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeatureReconstructionModel, self).__init__()
        # 使用一个简单的解码器将后层特征映射回前层特征空间
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 假设前层特征是隐藏维度
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)   # 重建为前层特征的维度
        )

    def forward(self, x):
        return self.decoder(x)

def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, keep_ratio: float, global_index: torch.Tensor, box_mask_z: torch.Tensor):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    Args:
        attn (torch.Tensor): [B, num_heads, L_t + L_s, L_t + L_s], attention weights
        tokens (torch.Tensor):  [B, L_t + L_s, C], template and search region tokens
        lens_t (int): length of template
        keep_ratio (float): keep ratio of search region tokens (candidates)
        global_index (torch.Tensor): global index of search region tokens
        box_mask_z (torch.Tensor): template mask used to accumulate attention weights

    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
        keep_index (torch.Tensor): indices of kept search region tokens
        removed_index (torch.Tensor): indices of removed search region tokens
    """
    lens_s = attn.shape[-1] - lens_t    
    bs, hn, _, _ = attn.shape

    lens_keep = math.ceil(keep_ratio * lens_s)
    if lens_keep == lens_s:
        return tokens, global_index, None

    attn_t = attn[:, :, :lens_t, lens_t:]

    


    if box_mask_z is not None:
        #print("\n1\n1\n1")
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        # attn_t = attn_t[:, :, box_mask_z, :]
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

        # attn_t = [attn_t[i, :, box_mask_z[i, :], :] for i in range(attn_t.size(0))]
        # attn_t = [attn_t[i].mean(dim=1).mean(dim=0) for i in range(len(attn_t))]
        # attn_t = torch.stack(attn_t, dim=0)
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)



    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]
    
    keep_index = global_index.gather(dim=1, index=topk_idx)
    
    removed_index = global_index.gather(dim=1, index=non_topk_idx)
    

    # separate template and search tokens
    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_s.shape
    # topk_idx_ = topk_idx.unsqueeze(-1).expand(B, lens_keep, C)

    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    # inattentive_tokens = tokens_s.gather(dim=1, index=non_topk_idx.unsqueeze(-1).expand(B, -1, C))

    # compute the weighted combination of inattentive tokens
    # fused_token = non_topk_attn @ inattentive_tokens
    
    # concatenate these tokens
    # tokens_new = torch.cat([tokens_t, attentive_tokens, fused_token], dim=0)
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

    #print("finish ce func")

    return tokens_new, keep_index, removed_index                       # x, global_index_search, removed_index_search


class CEABlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)     #from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

        self.keep_ratio_search = keep_ratio_search


        # self.adap_t = Bi_direct_adapter()        
        # self.adap2_t = Bi_direct_adapter()
        # self.adap_cross = CrossModal_Templates_Update(dim)
        # self.adap_cross_template = CrossAttention(dim)
        # self.adap_cross_template2 = CrossAttention(dim)
        # self.adap_norm3 = norm_layer(dim)        


    def forward(self, x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask=None, ce_template_mask=None, keep_ratio_search=None,dynamic_template=None,Test=None):
        #for two templates
        #准备两个模板用于计算注意力学习
        # z = x[:, :64]
        # zi = xi[:,:64]
        # new_x = x[:,64:]
        # new_xi = xi[:,64:]
        # z_ori = z
        # zi_ori = zi
        # if dynamic_template ==None:
        #     z_d,zi_d = z_ori,zi_ori
        # else:
        #     z_d,zi_d = torch.chunk(dynamic_template,2,dim=1)

        # new_z = z.clone()
        # new_zi = zi.clone()
        # for i in range(0,new_z.shape[0]-1):
        #     new_z[i] = z[i+1]
        #     new_zi[i] = zi[i+1]
        # new_z[-1] = z[0]
        # new_zi[-1] = zi[0]
        # # z = new_z + z
        # # zi = new_zi + zi
        # # z = z_ori+self.adap_cross_template(self.adap_norm3(new_z),self.adap_norm3(z))
        # # new_z = new_z + self.adap_cross_template(self.adap_norm3(z),self.adap_norm3(new_z))
        # # zi = zi_ori+self.adap_cross_template(self.adap_norm3(new_zi),self.adap_norm3(zi))
        # #20240701
        # # new_zi = new_zi + self.adap_cross_template((zi),(new_zi))
        # #20240702
        # # new_zi = new_zi + self.adap_cross_template(self.adap_norm3(zi),self.adap_norm3(new_zi))
        # z = torch.cat([z,z_d],dim=1)
        # zi = torch.cat([zi,zi_d],dim=1)
        # # z = torch.cat([new_z,z],dim=-1)
        # # zi = torch.cat([new_zi,zi],dim=-1)
        # # z = z_ori + self.adap_fusion(z)
        # # zi = zi_ori + self.adap_fusion(zi)
        # # z = z_ori + z
        # # zi = zi_ori + zi
        # x = torch.cat([z,new_x],dim=1)
        # xi = torch.cat([zi,new_xi],dim=1)


       


        xori = x
        
        x_attn, attn = self.attn(self.norm1(x), mask, True)   
        x = x + self.drop_path(x_attn)
        # x = x + self.drop_path(x_attn) + self.drop_path(self.adap_t(self.norm1(xi)))  #########-------------------------adapter

        xi_attn, i_attn = self.attn(self.norm1(xi), mask,True)
        xi = xi + self.drop_path(xi_attn)
        # xi = xi + self.drop_path(xi_attn) + self.drop_path(self.adap_t(self.norm1(xori)))  #########-------------------------adapter
                     
        lens_t = global_index_template.shape[1]

        removed_index_search = None
        removed_index_searchi = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
            xi, global_index_searchi, removed_index_searchi = candidate_elimination(i_attn, xi, lens_t, keep_ratio_search, global_index_searchi, ce_template_mask)

        xori = x
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adap2_t(self.norm2(xi)))   ###-------adapter

        xi = xi + self.drop_path(self.mlp(self.norm2(xi)))
        # xi = xi + self.drop_path(self.mlp(self.norm2(xi))) + self.drop_path(self.adap2_t(self.norm2(xori)))   ###-------adapter
        # x,xi,rgb_att,tir_att = self.adap_cross(x,xi)

        # print('x',x.size())
        #恢复原始形状
        # z = x[:, :64]
        # zi = xi[:,:64]
        # d_tokens = x[:,64:128]
        # di_tokens = xi[:,64:128]
        # dynamic_template = torch.cat([d_tokens,di_tokens],dim=1)
        # new_x = x[:,128:]
        # new_xi = xi[:,128:]
        # x = torch.cat([z,new_x],dim=1)
        # xi = torch.cat([zi,new_xi],dim=1)       

        return x, global_index_template, global_index_search, removed_index_search, attn, xi,global_index_templatei, global_index_searchi, removed_index_searchi, i_attn

class CEABlock_Enhancement(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self.attn = Attention_Uncertainty(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)     #from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

        self.keep_ratio_search = keep_ratio_search

        
        # self.adap_t = Bi_direct_adapter()        
        # self.adap2_t = Bi_direct_adapter()
        self.adap_fusion = CrossModal_ST_Fusion_with_uncertainty(dim)
        self.adap_fusion2 = CrossModal_ST_Fusion_with_uncertainty(dim)
        # self.adap_attn = Attention_Uncertainty_Enhancement(dim)
        # self.adap_attn2 = Attention_Uncertainty_Enhancement(dim)
        # self.adap_enhancement = EfficientAdditiveAttention(dim,dim)
        # self.adap_enhancement2 = EfficientAdditiveAttention(dim,dim)


    def forward(self, x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask=None, ce_template_mask=None, keep_ratio_search=None,Test=None,dynamic_template=None,template_masks=None):

        xori = x

        v,vi,T,[u_m,u,ui],alpha_m1 = self.adap_fusion(self.norm1(x),self.norm1(xi)) 


        # v,vi = self.adap_attn(self.norm1(x),self.norm1(xi))    
        x_attn, attn = self.attn(self.norm1(x), mask, True)   
        x = x + self.drop_path(x_attn)+self.drop_path(vi)
        # x = x + self.drop_path(x_attn) + self.drop_path(self.adap_t(self.norm1(xi)))  #########-------------------------adapter

        xi_attn, i_attn = self.attn(self.norm1(xi), mask,True)
        xi = xi + self.drop_path(xi_attn)+self.drop_path(v) 
        # xi = xi + self.drop_path(xi_attn) + self.drop_path(self.adap_t(self.norm1(xori)))  #########-------------------------adapter

        # x_fusion = self.adap_fusion(self.norm1(x),self.norm1(xi))
        # x = x + self.drop_path(x_fusion)
        # xi = xi + self.drop_path(x_fusion)
                     
        lens_t = global_index_template.shape[1]

        removed_index_search = None
        removed_index_searchi = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
            xi, global_index_searchi, removed_index_searchi = candidate_elimination(i_attn, xi, lens_t, keep_ratio_search, global_index_searchi, ce_template_mask)

        xori = x
        # x_e = self.adap_enhancement2(self.norm2(x),template_masks)
        # xi_e = self.adap_enhancement2(self.norm2(xi),template_masks)


        v,vi,T,[u_m,u,ui],alpha_m2 = self.adap_fusion2(self.norm2(x),self.norm2(xi))



        # v,vi = self.adap_attn2(self.norm2(x),self.norm2(xi))
        x = x + self.drop_path(self.mlp(self.norm2(x)))+self.drop_path(vi) 
        # x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adap2_t(self.norm2(xi)))   ###-------adapter

        xi = xi + self.drop_path(self.mlp(self.norm2(xi)))+self.drop_path(v)
        # xi = xi + self.drop_path(self.mlp(self.norm2(xi))) + self.drop_path(self.adap2_t(self.norm2(xori)))   ###-------adapter
        # x_fusion = self.adap_fusion2(self.norm2(x),self.norm2(xi))
        # x = x + self.drop_path(x_fusion)
        # xi = xi + self.drop_path(x_fusion)
 

        return x, global_index_template, global_index_search, removed_index_search, attn, xi,global_index_templatei, global_index_searchi, removed_index_searchi, i_attn,alpha_m1,alpha_m2,[u_m,u,ui]



class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        #print("class Block ")
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        #print("class Block forward")
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x