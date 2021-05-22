#test_pretrained_model
import os
import torch
import torchvision
from torch.nn.parameter import Parameter
from module.net import * # Generator,Mapping
import module.EAE_model.BE_v2 as BE
import module.EAE_model.D2E_v3 as BE2


def set_seed(seed): #随机数设置
    np.random.seed(seed)
    #random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True

set_seed(0)

Gs = Generator(startf=64, maxf=512, layer_count=7, latent_size=512, channels=3)
Gs.load_state_dict(torch.load('/Users/apple/Desktop/my-Style/model-result-v1/pre-model/bedroom/bedrooms256_Gs_dict.pth',map_location=torch.device('cpu')))
Gm = Mapping(num_layers=14, mapping_layers=8, latent_size=512, dlatent_size=512, mapping_fmaps=512) #num_layers: 14->256 / 16->512 / 18->1024
Gm.load_state_dict(torch.load('/Users/apple/Desktop/my-Style/model-result-v1/pre-model/bedroom/bedrooms256_Gm_dict.pth',map_location=torch.device('cpu')))
Gm.buffer1 = torch.load('/Users/apple/Desktop/my-Style/model-result-v1/pre-model/bedroom/bedrooms256_tensor.pt')

layer_num = 14 # 14->256 / 16 -> 512  / 18->1024 
layer_idx = torch.arange(layer_num)[np.newaxis, :, np.newaxis] # shape:[1,18,1], layer_idx = [0,1,2,3,4,5,6。。。，17]
ones = torch.ones(layer_idx.shape, dtype=torch.float32) # shape:[1,18,1], ones = [1,1,1,1,1,1,1,1]
coefs = torch.where(layer_idx < layer_num//2, 0.7 * ones, ones) # 18个变量前8个裁剪比例truncation_psi [0.7,0.7,...,1,1,1] 

E1 = BE.BE(startf=64, maxf=512, layer_count=7, latent_size=512, channels=3)
E1.load_state_dict(torch.load('/Users/apple/Desktop/my-Style/model-result-v1/D2E/Ev2_bedroom256_ep115000.pth',map_location=torch.device('cpu')))

E2 = BE2.BE(startf=64, maxf=512, layer_count=7, latent_size=512, channels=3)
E2.load_state_dict(torch.load('/Users/apple/Desktop/my-Style/model-result-v1/D2E/Ev2_bedroom256_ep115000.pth',map_location=torch.device('cpu')),strict=False)


# print('------------------')
# for param1,param2 in zip(E1.parameters(),E2.parameters()):
#     print(param1 == param2)

print(E1)
print('------------------')
print(E2)

# #inference
# batch_size=5
# with torch.no_grad():
#     latents = torch.randn(batch_size, 512)
#     w1 = Gm(latents,coefs_m=coefs)

#     imgs1 = Gs.forward(w1,6)
#     const2,w2 = E2(imgs1)
#     Gs.const = Parameter(const2)
#     imgs2 = Gs.forward(w2,6)
#     test_img = torch.cat((imgs1[:batch_size],imgs2[:batch_size]))*0.5+0.5
#     torchvision.utils.save_image(test_img, './test_modelV3.png',nrow=batch_size)

