import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossAttention, self).__init__()
        self.linear_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = hidden_dim ** -0.5
       
        self.linear_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
    
    def forward(self, T, S):
        q = self.linear_q(T)  # query
        k = self.linear_k(S)  # key
        v = self.linear_v(S)  # value
        
        # 计算注意力权重
        attn_weights = torch.matmul(q, k.transpose(-2, -1))* self.scale
        attn_weights = self.softmax(attn_weights)
        
        # 使用注意力权重加权求和
        attn_output = torch.matmul(attn_weights, v)
        
        # 输出结果
        output = self.linear_out(attn_output)
        return output
    
class CrossAttention_For_Fusion_uncertainty(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossAttention_For_Fusion_uncertainty, self).__init__()
        self.linear_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # self.linear_ki = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # self.linear_vi = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.ReLU = nn.ReLU()

        self.Softplus = nn.Softplus()        
        self.scale = hidden_dim ** -0.5
        self.proj_drop = nn.Dropout(0.1)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        # self.norm2 = nn.LayerNorm(hidden_dim)
        self.len_t = 64
    
    def forward(self, T, x,xi):
        B,_,_ = T.shape
        x_ori = x
        xi_ori = xi
        x = x[:,self.len_t:]
        xi = xi[:,self.len_t:]
        q = self.linear_q(T)  # query
        k = self.linear_k(x)  # key
        ki = self.linear_k(xi)  # key
        v = self.linear_v(x)
        vi = self.linear_v(xi)  # value

        # print(self.linear_v.weight)
        
        # 计算注意力权重
        attn_weights = torch.matmul(q, k.transpose(-2, -1))* self.scale
        #attn_weights torch.Size([32, 64, 256])
        # attn_weights = self.softmax(attn_weights)
        #attn_weightsi torch.Size([32, 64, 192])
        attn_weightsi = torch.matmul(q, ki.transpose(-2, -1))* self.scale
        # attn_weightsi = self.softmax(attn_weightsi)

        # 在列的维度（dim=1）上求和，并保持该维度
        evidence = torch.mean(attn_weights, dim=1, keepdim=True)
        evidencei = torch.mean(attn_weightsi, dim=1, keepdim=True)  
        alpha = self.Softplus(evidence) + 1
        alphai = self.Softplus(evidencei) + 1
        S = torch.sum(alpha,dim=-1)
        Si = torch.sum(alphai,dim=-1)


        b = evidence / S
        bi = evidencei / Si
        u = 256 / S
        ui = 256 / Si

        b_m = (b*ui + bi*u) / (u+ui)
        u_m = (2*u*ui) / (u+ui)
        alpha_m = (alpha + alphai) / 2
        S_m = torch.sum(alpha_m,dim=-1)

        weights_rgb = alpha / S.unsqueeze(-1).expand(evidence.shape)
        weights_tir = alphai / Si.unsqueeze(-1).expand(evidencei.shape)


        
        weights = alpha_m / S_m.unsqueeze(-1).expand(evidencei.shape)
        # weights_u = weights_u.squeeze(1)

        min_v = weights.min()
        max_v = weights.max()
        # 使用 min-max 归一化将值缩放到 0.01 到 1 之间
        weights = 0.0001 + (weights - min_v) / (max_v - min_v+0.0001) * (1 - 0.01)     
        # weights =  0.01 + (weights - min_v) / (max_v - min_v) * (1 - 0.01)                 
        # u_m = torch.squeeze(256 / S_m)

        #利用不确定性作为权重
        u_r_weight = u_m/u
        u_i_weight = u_m/ui
       

        v = v * weights.permute(0,2,1)*(u_r_weight.unsqueeze(-1))
        vi = vi * weights.permute(0,2,1)*(u_i_weight.unsqueeze(-1))

        # x_s = v+vi
        # # x_s = v*(1-u.permute(0,2,1)) + vi*(1-ui.permute(0,2,1))

        # #for enhance target features
        # x_s = self.linear_out(x_s)
        # x_s = self.proj_drop(x_s)

        v = self.linear_out(v)
        v = self.proj_drop(v)
        vi = self.linear_out(vi)
        vi = self.proj_drop(vi)
        # T = T + x_output + xi_output
        # T = self.norm(T)


        # # 输出结果
        # output = self.linear_out(attn_output)
        return v,vi,T,[u_m,u,ui],alpha_m

class CrossModal_ST_Fusion_with_uncertainty(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossModal_ST_Fusion_with_uncertainty,self).__init__()
        self.adap_cross = CrossAttention_For_Fusion_uncertainty(hidden_dim=768)
        self.len_t = 64

        # self.adap_down = nn.Linear(1536, 768)
        self.adap_linear = nn.Linear(768, 768)
        self.r = 0.5 #setting for penalty intensity parameter   
        self.dzd_dropout = nn.Dropout(p=0.3)
        # self.adap_linear = nn.Linear(768,768)
        # 极大池化层
        # self.max_pool = nn.MaxPool2d(kernel_size=4, stride=4)
        # self.norm1 = nn.LayerNorm(hidden_dim)
        # self.norm2 = nn.LayerNorm(hidden_dim)
        # self.relu = nn.ReLU()


    def Uncertainty_Measure(self,x):
        num_samples=10
        features = []
        for i in range(num_samples):
            x_ = self.dzd_dropout(x)
            features.append(x_)
        features = torch.stack(features)
        mean_x = features.mean(0)
        std_x = features.std(0).mean(-1)
        return mean_x,std_x

    # def forward(self,x,x_z,xi,xi_z):
    def forward(self,x,xi,template_mask=None):
        B,_,_ = x.shape
        z = x[:, :self.len_t]
        x_ = x[:, self.len_t:]
        zi = xi[:, :self.len_t]
        xi_ = xi[:, self.len_t:]   
        
        # if template_mask is not None:
        #     self.template_mask = template_mask
        #     z = z * self.template_mask.unsqueeze(-1)
        #     zi = zi * self.template_mask.unsqueeze(-1)
        
        mean_z,std_z = self.Uncertainty_Measure(z)
        mean_zi,std_zi = self.Uncertainty_Measure(zi)
        C_r = torch.exp(-std_z*self.r)
        C_t = torch.exp(-std_zi*self.r) 
        # U_r = 1-C_r.squeeze(-1).mean(dim=1)
        # U_t = 1-C_t.squeeze(-1).mean(dim=1)       
        # print(x.size())
        # z = torch.cat((z,zi),dim=-1)
        z = mean_z * C_r.unsqueeze(-1) + mean_zi*C_t.unsqueeze(-1)
        # print('z1',z.size())

        

        z = self.adap_linear(z)
        


        # print('z2',z.size())
        x_output,xi_output,T,[u_m,u,ui],weights = self.adap_cross(z,x,xi)
        # output = self.norm1(output)
        # outputi = self.norm2(outputi)
        # x = torch.cat([output,x_],dim=1)
        # xi = torch.cat([outputi,xi_],dim=1)
        T = T[:,:64]
        x_output = torch.cat((T,x_output),dim=1)
        xi_output = torch.cat((T,xi_output),dim=1)
        return x_output,xi_output,T,[u_m,u,ui],weights

class Spatio_Temporal_Uint(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.dim = hidden_dim
        self.linear_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.ReLU = nn.ReLU()

        self.Softplus = nn.Softplus()        
        self.scale = hidden_dim ** -0.5
        self.proj_drop = nn.Dropout(0.1)
        self.linear_down = nn.Linear(2*hidden_dim, hidden_dim, bias=False)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # self.norm = nn.LayerNorm(hidden_dim)
        #实现简单的空间注意力机制 增强目标特征
        self.spatio_attn = nn.Linear(hidden_dim, 1) 



    def forward(self,t,memory):
        B, N, C = t.shape
        self.tk = self.linear_k(t)
        self.memoryk = self.linear_q(memory)
        self.t = self.linear_v(t)
        self.memory = self.linear_v(memory)

        attn_weights = torch.matmul(self.tk, self.memoryk.transpose(-2, -1))* self.scale
        attn_weights = self.softmax(attn_weights)
        # T = attn_weights @ self.t
        Memories = attn_weights.transpose(-2, -1) @ self.memory
        # T = self.linear_out(T)
        Memories = self.proj_drop(Memories)
        Memories = self.linear_out(Memories)
        #利用空间注意力增强目标特征
        # T_w = self.spatio_attn(self.t)
        # T_w = torch.sigmoid(T_w)
        # T_w = T_w.expand(-1,-1,t.size(-1))
        Memories_w = self.spatio_attn(Memories)
        Memories_w = torch.sigmoid(Memories_w)
        Memories_w = Memories_w.expand(-1,-1,t.size(-1))
        F = Memories
        # Memories = Memories_w*Memories_w

        # F_SUM = torch.cat([T,Memories],dim=-1)
        # F = self.linear_down(F_SUM)

        return F



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
class BasicBlock(nn.Module):
    
    def __init__(self, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__() 

        self.adap_fusion = nn.Linear(planes*2,planes)
        self.cat = ChannelAttention(planes)
        self.sat = SpatialAttention() 
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention() 
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x,xi):
        B,L,C = x.shape
        Template,S = x[:,:64],x[:,64:]
        Templatei,Si = xi[:,:64],xi[:,64:]
        Template = Template
        Templatei = Templatei
        S = S.view(B,C,16,16)
        Si = Si.view(B,C,16,16)

        T = self.adap_fusion(torch.cat([Template,Templatei],dim=-1)).view(B,C,8,8)
        #for the enhancement of template
        c_wt = self.cat(T)
        outT = c_wt*T
        s_wt = self.sat(outT)
        outT = s_wt*outT

        #for the enhancement of search
        c_w = self.ca(S)
        outS = c_w*S
        s_w = self.sa(outS)
        outS = s_w*outS

        c_wi = self.ca(Si)
        outSi = c_wi*Si
        s_wi = self.sa(outSi)
        outSi = s_wi*outSi

        # outS = outS + S
        # outSi = outSi + Si
        # T = outT + T

        outS = self.relu(outS).view(B,256,C)
        outSi = self.relu(outSi).view(B,256,C)
        outT = self.relu(outT).view(B,64,C)

        outX = torch.cat([outT,outS],dim=1)
        outXi = torch.cat([outT,outSi],dim=1)

        return outX,outXi,s_wt,s_w,s_wi,c_w,c_wi



class TemplateRouter(nn.Module):
    def __init__(self, L, C):
        super(TemplateRouter, self).__init__()
        # 假设我们用全局平均池化来压缩每个模板的特征
        self.gap = nn.AdaptiveAvgPool2d(1)  # GAP池化到1x1
        
        # 为每个模板设计一个简单的全连接层，用于计算评分
        self.fc = nn.Sequential(
            nn.Linear(C, 64),  # 假设 C 是输入通道数，输出一个64维的特征
            nn.ReLU(),
            nn.Linear(64, 1)   # 输出一个标量评分
        )
        
        # 用于归一化得分的Softmax
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, template1, template2):
        """
        :param template1: [B, L, C]
        :param template2: [B, L, C]
        :return: [B, 2] 评分张量，和相对强弱的归一化评分
        """
        # 对输入模板进行变换，先调整成 [B, C, L] 形式
        template1 = template1.permute(0, 2, 1)  # [B, C, L]
        template2 = template2.permute(0, 2, 1)  # [B, C, L]
        
        # 使用全局平均池化得到 [B, C, 1]
        template1 = self.gap(template1.unsqueeze(2))  # 增加一个维度 [B, C, 1, 1]
        template2 = self.gap(template2.unsqueeze(2))  # 增加一个维度 [B, C, 1, 1]
        
        # 拉平成 [B, C] 形状
        template1 = template1.view(template1.size(0), -1)  # [B, C]
        template2 = template2.view(template2.size(0), -1)  # [B, C]
        
        # 计算每个模板的评分
        score1 = self.fc(template1)  # [B, 1]
        score2 = self.fc(template2)  # [B, 1]
        
        # 将评分合并成一个 [B, 2] 的张量
        scores = torch.cat([score1, score2], dim=1)  # [B, 2]
        
        # 使用 Softmax 进行归一化，得到相对强弱的概率分布
        relative_score = self.softmax(scores)  # [B, 2]
        
        return scores, relative_score