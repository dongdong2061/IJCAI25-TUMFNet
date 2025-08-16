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



class SK_Fusion(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=2,r=16,L=32):
        '''
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param stride:  步长，默认为1
        :param M:  分支数
        :param r: 特征Z的长度，计算其维度d 时所需的比率（论文中 特征S->Z 是降维，故需要规定 降维的下界）
        :param L:  论文中规定特征Z的下界，默认为32
        采用分组卷积： groups = 32,所以输入channel的数值必须是group的整数倍
        '''
        super(SK_Fusion,self).__init__()
        d=max(in_channels//r,L)   # 计算从向量C降维到 向量Z 的长度d
        self.M=M
        self.out_channels=out_channels
        self.global_pool=nn.AdaptiveAvgPool2d(output_size = 1) # 自适应pool到指定维度    这里指定为1，实现 GAP
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.fc1=nn.Sequential(nn.Conv2d(out_channels,d,1,bias=False),
                            #    nn.BatchNorm2d(d),
                               nn.ReLU(inplace=True))   # 降维
        self.fc2=nn.Conv2d(d,out_channels*M,1,1,bias=False)  # 升维
        self.softmax=nn.Softmax(dim=1) # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1
    def forward(self, input):
        batch_size=input[0].size(0)
        output=input

        output[0] = self.conv1(input[0].reshape(batch_size,768,8,8))
        output[1] = self.conv2(input[1].reshape(batch_size,768,8,8))
        #the part of fusion
        U=reduce(lambda x,y:x+y,output) # 逐元素相加生成 混合特征U  [batch_size,channel,H,W]
        # print(U.size())            
        s=self.global_pool(U)     # [batch_size,channel,1,1]
        # print(s.size())
        z=self.fc1(s)  # S->Z降维   # [batch_size,d,1,1]
        # print(z.size())
        a_b=self.fc2(z) # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b   [batch_size,out_channels*M,1,1]
        # print(a_b.size())
        a_b=a_b.reshape(batch_size,self.M,self.out_channels,-1) #调整形状，变为 两个全连接层的值[batch_size,M,out_channels,1]  
        # print(a_b.size())
        a_b=self.softmax(a_b) # 使得两个全连接层对应位置进行softmax [batch_size,M,out_channels,1]  
        #the part of selection
        a_b=list(a_b.chunk(self.M,dim=1))#split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块 [[batch_size,1,out_channels,1],[batch_size,1,out_channels,1]
        # print(a_b[0].size())
        # print(a_b[1].size())
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b)) # 将所有分块  调整形状，即扩展两维  [[batch_size,out_channels,1,1],[batch_size,out_channels,1,1]
        V=list(map(lambda x,y:x*y,output,a_b)) # 权重与对应  不同卷积核输出的U 逐元素相乘[batch_size,out_channels,H,W] * [batch_size,out_channels,1,1] = [batch_size,out_channels,H,W]
        V=reduce(lambda x,y:x+y,V) # 两个加权后的特征 逐元素相加  [batch_size,out_channels,H,W] + [batch_size,out_channels,H,W] = [batch_size,out_channels,H,W]
        # print("V",V.size())
        V = V.reshape(batch_size,-1,self.out_channels,)  # [batch_size,out_channels,H,W] -> [batch_size,out_channels,H*W]
        # print("V",V.size())
        return V    # [batch_size,out_channels,H,W]

class Spatio_Temporal_Mine(nn.Module):
    """
    to mine spatio-temporal information among consecutive frames
    """
    def __init__(self,dim):
        super().__init__()
        self.q = nn.Linear(dim,dim)
        self.k = nn.Linear(dim,dim)
        self.v = nn.Linear(dim,dim)
        self.softmax = nn.Softmax(dim=-1)
        self.Softplus = nn.Softplus()        
        self.scale = dim ** -0.5
        self.proj_drop = nn.Dropout(0.1)
        self.linear_out = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)


        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=2, padding=1)
        self.conv1_1 = nn.Linear(2*dim,dim)
        # self.relu = nn.ReLU()
        self.conv2 = nn.Linear(2*dim,dim)
        self.conv3 = nn.Linear(3*dim,dim)
        self.norm = nn.LayerNorm(dim)

     

    def forward(self,x,xi,query):
        """
        x:tensor[B,320,768]
        """
        x_s = x[:,64:]
        B,N,D= x_s.shape
        # x_s = x_s.view(B,D,16,16)
        # x_conv1 = self.relu(self.conv1(x_s))
        # x_conv1 = x_conv1.view(B,8,8,D).reshape(B,64,D)
        q = self.q(query)
        k = self.k(x_s)
        v = self.v(x_s)
        xi_s = xi[:,64:]
        B,N,D= xi_s.shape
        # xi_s = xi_s.view(B,D,16,16)
        # xi_conv1 = self.relu(self.conv1(xi_s))
        # xi_conv1 = xi_conv1.view(B,8,8,D).reshape(B,64,D)
        ki = self.k(xi_s)
        vi = self.v(xi_s)

        # 计算注意力权重
        attn_weights = torch.matmul(q, k.transpose(-2, -1))* self.scale 
        attn_weightsi = torch.matmul(q, ki.transpose(-2, -1))* self.scale 
        attn_e = attn_weights 
        attn_ei = attn_weightsi
        attn_weights = self.softmax(attn_weights)
        attn_weightsi = self.softmax(attn_weightsi)

        # 在列的维度（dim=1）上求和，并保持该维度
        evidence = torch.mean(attn_e, dim=1, keepdim=True)
        evidencei = torch.mean(attn_ei, dim=1, keepdim=True)
        # alpha = self.Softplus(evidence) + 1
        alpha = self.Softplus(evidence) + 1
        alphai = self.Softplus(evidencei) + 1
        S = torch.sum(alpha,dim=-1).unsqueeze(-1)
        Si = torch.sum(alphai,dim=-1).unsqueeze(-1)
        b = evidence / S
        bi = evidencei / Si
        u = 256 / S
        ui = 256 / Si

        b_m = (b*ui + bi*u) / (u+ui)
        u_m = (2*u*ui) / (u+ui)
        alpha_m = (alpha + alphai) / 2
        S_m = torch.sum(alpha_m,dim=-1)

        weightsr = alpha / S.expand(evidence.shape)
        weightsi = alphai / S.expand(evidencei.shape)
        weights = alpha_m / S_m.expand(evidencei.shape)

        min_v = weights.min()
        max_v = weights.max()
        # 使用 min-max 归一化将值缩放到 0.01 到 1 之间
        weights = 0.0001 + (weights - min_v) / (max_v - min_v+0.0001) * (1 - 0.01) 

        #利用不确定性作为权重
        u_r_weight = u_m/u
        u_i_weight = u_m/ui
        x_conv1_res = v * weights.squeeze(1).unsqueeze(-1)*(u_r_weight.unsqueeze(-1))
        xi_conv1_res = vi * weights.squeeze(1).unsqueeze(-1)*(u_i_weight.unsqueeze(-1))
        x_f = torch.cat([xi_conv1_res,x_conv1_res],dim=-1)
        #搜索区域信息聚合
        x_f = self.conv1_1(x_f)
        x_f = self.conv1(x_f.view(B,D,16,16)).reshape(B,64,D)
        #交叉注意力聚合信息 多模态融合
        # 使用注意力权重加权求和
        attn_output = torch.matmul(attn_weights, x_conv1_res)
        attn_outputi = torch.matmul(attn_weightsi, xi_conv1_res)         
        # 输出结果
        output = self.linear_out(attn_output)
        output = self.proj_drop(output)
        outputi = self.linear_out(attn_outputi)
        outputi = self.proj_drop(outputi)
        # output_f = output + outputi
        output_f = torch.cat([output,outputi],dim=-1)
        output_f = self.conv2(output_f).squeeze(0)   
        # print(output_f.size(),x_f.size(),q.size())
        #聚集搜索区域信息  模板信息   交叉注意力聚合信息
        x_final = torch.cat([x_f,q,output_f],dim=-1)
        x_final = self.conv3(x_final)
        # # tensor[B,64,768]
        Queries = x_final
        Queries = self.norm(output_f + x_f + q)

        return Queries,alpha_m

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


    def add_gaussian_noise_with_prob(self,feature, mean=0.0, std=0.1, prob=0.5):
        """
        以指定概率向特征添加高斯噪声
        Args:
            feature (torch.Tensor): 输入特征，形状为 [1, 320, 768]
            mean (float): 高斯噪声的均值，默认为0
            std (float): 高斯噪声的标准差，控制噪声强度，默认为0.1
            prob (float): 添加噪声的概率，默认为50%
        Returns:
            torch.Tensor: 添加噪声后的特征（或原始特征）
        """
        if random.random() < prob:  # 按概率触发
            noise = torch.randn_like(feature) * std + mean  # 生成相同形状的高斯噪声
            noisy_feature = feature + noise
            return noisy_feature
        else:
            return feature  # 不添加噪声


        # self.adap_selfattn_uncerttainty1 = Attention_Uncertainty_Enhancement(embed_dim)
        # self.adap_selfattn_uncerttainty2 = Attention_Uncertainty_Enhancement(embed_dim)
        # self.adap_selfattn_uncerttainty3 = Attention_Uncertainty_Enhancement(embed_dim)
        # self.uncertainty_fusion = CrossModal_ST_Fusion_with_uncertainty(embed_dim)
        # self.adap_fusion_SK = SK_Fusion(embed_dim,embed_dim)
        # self.adap_fusion_pooling = nn.MaxPool2d(kernel_size=3,stride=2)
        # self.adap_fusion_conv = nn.Linear(embed_dim*2,embed_dim)

        # self.adap_conv = nn.Linear(embed_dim*2,embed_dim)

        # self.adap_MSTE = CEABlock_Spatio_Temporal(
        #                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
        #                 attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
        #                 keep_ratio_search=ce_keep_ratio_i)
        # self.adap_norm2 = norm_layer(embed_dim)        
        # self.adapter_head = Bi_appearence_adapter()
        # self.adap_fusion = CrossModal_ST_Fusion(embed_dim)
        # self.adap_crossstage = Cross_Stage_attention(embed_dim)
        # self.adap_crossstage = Cross_Stage_Linear(embed_dim)
        # self.adap_fusion  = nn.Linear(1536,768)
    def pearson_correlation(self,tensor_a, tensor_b):
        """
        计算两个 token 向量之间的皮尔逊相关系数。
        
        参数：
        - tensor_a (Tensor): 第一个特征 token，形状为 [B, N]。
        - tensor_b (Tensor): 第二个特征 token，形状为 [B, N]。
        
        返回：
        - correlation (Tensor): 皮尔逊相关系数，形状为 [B]，每个元素是对应样本的相关系数。
        """
        # 计算均值
        mean_a = tensor_a.mean(dim=-1, keepdim=True)
        mean_b = tensor_b.mean(dim=-1, keepdim=True)
        
        # 计算协方差
        cov_ab = ((tensor_a - mean_a) * (tensor_b - mean_b)).mean(dim=-1)
        
        # 计算标准差
        std_a = tensor_a.std(dim=-1)
        std_b = tensor_b.std(dim=-1)
        
        # 计算皮尔逊相关系数
        correlation = cov_ab / (std_a * std_b)
        return correlation
    
    def get_topk(self,input_tensor, k, n):
        """
        生成掩码：如果 token 中大于阈值 k 的元素数量超过 n，则将前 n 个最大值位置设为 1，
        否则将所有大于 k 的元素的位置设为 1，其余位置为 0。

        参数：
        - input_tensor (Tensor): 输入张量，形状为 [B, 256, 768]，其中 B 是 batch 大小。
        - k (float): 阈值，用于筛选大于 k 的元素。
        - n (int): 如果大于 k 的元素数量超过 n，则选择前 n 个最大值的索引。

        返回：
        - mask_tensor (Tensor): 生成的掩码，形状为 [B, 256, 768]，包含 0 和 1。
        """
        B, num_tokens, _ = input_tensor.shape  # 获取 batch 大小、token 数量和特征维度
        
        # 通过比较每个 token 中的每个特征是否大于 k，生成布尔掩码
        mask = input_tensor/256 > k  # 形状为 [B, 256, 768]
        
        # 创建一个全零的张量用于保存掩码结果
        mask_tensor = torch.zeros_like(input_tensor, dtype=torch.int)  # 形状为 [B, 256, 768]

        # 遍历 batch 中的每个样本
        for batch_idx in range(B):
            for token_idx in range(num_tokens):
                # 获取当前 token 中大于 k 的元素的布尔掩码
                token_mask = mask[batch_idx, token_idx]  # 形状为 [768]
                
                # 获取大于 k 的元素的索引
                indices = token_mask.nonzero(as_tuple=True)[0]  # 形状为 [N]，N 为大于 k 的元素数量
                
                # 判断当前 token 中大于 k 的数量是否大于 n
                if len(indices) > n:
                    # 如果大于 k 的元素数量大于 n，则选择前 n 个最大值的索引
                    token_values = input_tensor[batch_idx, token_idx]  # 当前 token 的值，形状为 [768]
                    
                    # 获取排序后的索引（按值从大到小排序）
                    sorted_indices = token_values.argsort(descending=True)  # 从大到小排序的索引
                    
                    # 选择前 n 个最大值的索引
                    selected_indices = sorted_indices[:n]
                    
                    # 将选择的索引位置的值设置为 1
                    mask_tensor[batch_idx, token_idx, selected_indices] = 1
                else:
                    # 否则，选择所有大于 k 的元素的索引
                    mask_tensor[batch_idx, token_idx, indices] = 1
        
        return mask_tensor 
    
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

        # 2. 添加噪声（50%概率）
        # result = 1 if random.random() < 0.5 else 0
        # if result == 1:
        #     result = 1 if random.random() < 0.5 else 0
        #     if result == 1:
        #         x = xi
        #     else:
        #         xi = x


        # # 2. 添加噪声（50%概率）
        # result = 1 if random.random() < 0.5 else 0
        # if result == 1:
        #     x = self.add_gaussian_noise_with_prob(x, std=10)
        # else:
        #     xi = self.add_gaussian_noise_with_prob(xi, std=10)


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
                for i in range(1,B):

                    z_new[0] = z[i].unsqueeze(0)
                    zi_new[0] = zi[i].unsqueeze(0)
                    z_new[0] = z[1].unsqueeze(0)
                    zi_new[0] = zi[1].unsqueeze(0)
                # if B == 4:
                #     zi_new[2] = zi[2].unsqueeze(0)
                #     z_new[2] = z[2].unsqueeze(0)
                #     z_new[3] = z[3].unsqueeze(0)
                #     zi_new[3] = zi[3].unsqueeze(0)
                #     z_new[0] = z[1].unsqueeze(0)
                #     zi_new[0] = zi[1].unsqueeze(0)
                #     z_new[1] = z[0].unsqueeze(0)
                #     zi_new[1] = zi[0].unsqueeze(0)
                # else:
                #     z_new[1] = z[-1].unsqueeze(0)
                #     zi_new[1] = zi[-1].unsqueeze(0)
                #     z_new[0] = z[1].unsqueeze(0)
                #     zi_new[0] = zi[1].unsqueeze(0)
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
