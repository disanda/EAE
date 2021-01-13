import torch
import torch.nn as nn 
from torch.nn import init
from torch.nn.parameter import Parameter
import sys
sys.path.append('../')
import module.lreq as ln
from module.net import Blur,FromRGB,downscale2d
from torch.nn import functional as F

# G 改 E, 实际上需要用G Block改出E block, 完成逆序对称，在同样位置还原style潜码
class BEBlock(nn.Module):
    def __init__(self, inputs, outputs, latent_size, has_last_conv=True, fused_scale=True): #分辨率大于128用fused_scale,即conv完成上采样
        super().__init__()
        self.has_last_conv = has_last_conv
        self.noise_weight_1 = nn.Parameter(torch.Tensor(1, inputs, 1, 1))
        self.noise_weight_1.data.zero_()
        self.bias_1 = nn.Parameter(torch.Tensor(1, inputs, 1, 1))
        self.instance_norm_1 = nn.InstanceNorm2d(inputs, affine=False, eps=1e-8)
        self.inver_mod1 = ln.Linear(2 * inputs, latent_size, gain=1) # [n, 2c] -> [n,512]
        self.conv_1 = ln.Conv2d(inputs, inputs, 3, 1, 1, bias=False)

        self.noise_weight_2 = nn.Parameter(torch.Tensor(1, inputs, 1, 1))
        self.noise_weight_2.data.zero_()
        self.bias_2 = nn.Parameter(torch.Tensor(1, inputs, 1, 1))
        self.instance_norm_2 = nn.InstanceNorm2d(inputs, affine=False, eps=1e-8)
        self.inver_mod2 = ln.Linear(2 * inputs, latent_size, gain=1)
        self.blur = Blur(inputs)
        if has_last_conv:
            if fused_scale:
                self.conv_2 = ln.Conv2d(inputs, outputs, 3, 2, 1, bias=False, transform_kernel=True)
            else:
                self.conv_2 = ln.Conv2d(inputs, outputs, 3, 1, 1, bias=False)

        self.fused_scale = fused_scale
        with torch.no_grad():
            self.bias_1.zero_()
            self.bias_2.zero_()

    def forward(self, x):
        mean1 = torch.mean(x, dim=[2, 3], keepdim=True) # [b, c, 1, 1]
        std1 = torch.sqrt(torch.mean((x - mean1) ** 2, dim=[2, 3], keepdim=True))  # [b, c, 1, 1]
        style1 = torch.cat((mean1, std1), dim=1) # [b,2c,1,1]
        w1 = self.inver_mod1(style1.view(style1.shape[0],style1.shape[1])) # [b,2c]->[b,512]

        x = self.instance_norm_1(x)
        x = F.leaky_relu(x, 0.2)
        x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_1, tensor2=torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]]).to(x.device))
        x = self.conv_1(x)

        mean2 = torch.mean(x, dim=[2, 3], keepdim=True) # [b, c, 1, 1]
        std2 = torch.sqrt(torch.mean((x - mean2) ** 2, dim=[2, 3], keepdim=True))  # [b, c, 1, 1]
        style2 = torch.cat((mean2, std2), dim=1) # [b,2c,1,1]
        w2 = self.inver_mod2(style2.view(style2.shape[0],style2.shape[1])) # [b,512] , 这里style2.view一直写错成style1.view

        x = self.instance_norm_2(x)
        x = F.leaky_relu(x, 0.2)
        x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_2, tensor2=torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]]).to(x.device))

        if self.has_last_conv:
            x = self.blur(x)
            x = self.conv_2(x)
            if not self.fused_scale: #在新的一层起初 fused_scale = flase, 完成上采样
                x = downscale2d(x)
        return x, w1, w2


class BE(nn.Module):
    def __init__(self, startf=16, maxf=512, layer_count=9, latent_size=512, channels=3):
        super().__init__()
        self.maxf = maxf
        self.startf = startf
        self.latent_size = latent_size
        self.layer_to_resolution = [0 for _ in range(layer_count)]
        self.decode_block = nn.ModuleList()
        inputs = startf # 16 
        outputs = startf*2
        resolution = 1024

        from_RGB = nn.ModuleList()
        for i in range(layer_count):

            has_last_conv = i+1 != layer_count
            fused_scale = resolution >= 128 # 在新的一层起初 fused_scale = flase, 完成上采样

            from_RGB.append(FromRGB(channels, inputs))
            block = BEBlock(inputs, outputs, latent_size, has_last_conv, fused_scale=fused_scale)

            inputs = inputs*2
            outputs = outputs*2
            inputs = min(maxf, inputs) 
            outputs = min(maxf, outputs)
            self.layer_to_resolution[i] = resolution
            resolution /=2
            self.decode_block.append(block)

        self.FromRGB = from_RGB

    #将w逆序，以保证和G的w顺序, block_num控制progressive
    def forward(self, x, block_num=9):
        x = self.FromRGB[9-block_num](x)
        w = torch.tensor(0)
        for i in range(9-block_num,9):
            x,w1,w2 = self.decode_block[i](x)
            w_ = torch.cat((w2.view(x.shape[0],1,512),w1.view(x.shape[0],1,512)),dim=1) # [b,2,512]
            if i == (9-block_num):
                w = w_ # [b,n,512]
            else:
                w = torch.cat((w_,w),dim=1)
            #print(w.shape)
        return x, w

    #和G的w逆序
    # def forward(self, x, block_num=9):
    #     x = self.FromRGB(x)
    #     w = torch.tensor(0)
    #     for i in range(9-block_num,9):
    #         x,w1,w2 = self.decode_block[i](x)
    #         w_ = torch.cat((w1.view(x.shape[0],1,512),w2.view(x.shape[0],1,512)),dim=1) # [b,2,512]
    #         if i == 0:
    #             w = w_ # [b,n,512]
    #         else:
    #             w = torch.cat((w,w_),dim=1)
    #         #print(w.shape)
    #     return x, w

# block1 [3, 3, 16, 16]
#        [3, 3, 16, 32] 1024
# block2 [3, 3, 32, 32]
#        [3, 3, 32, 64] 512
# block3 [3, 3, 64, 64]
#        [3, 3, 64  128] 256
# block4 [3, 3, 128, 128]
#        [3, 3, 128, 256] 128
# block5 [3, 3, 256, 256]
#        [3, 3, 256, 512] 64
# block6 [3, 3, 512, 512]
#        [3, 3, 512, 512] 32
# block7 [3, 3, 512, 512]
#        [3, 3, 512, 512] 16
# block8 [3, 3, 512, 512]
#        [3, 3, 512, 512] 8
# block9 [3, 3, 512, 512]
#        [3, 3, 512, 512] 4