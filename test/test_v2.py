import numpy as np
import torch
from configs.defaults import get_cfg_defaults
import argparse
from module.model import Model
from nativeUtils.checkpointer import Checkpointer
import logging
from torchvision.utils import save_image
from module.net import Generator, Mapping, Discriminator
import random
from torch.nn import init

#------------------随机数设置--------------
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True


#------------------配置文件------------------------
parser = argparse.ArgumentParser(description="StyleGAN")
parser.add_argument(
        "--config-file",
        default="configs/cat-bedroom-256.yaml",
        metavar="FILE",
        type=str,) # args.config_file
args = parser.parse_args()
cfg = get_cfg_defaults()
cfg.merge_from_file(args.config_file)
cfg.freeze()


#---------------------------------继承原版的Gs Gm , dlantent_avg
model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT, # startf = 16
        layer_count= cfg.MODEL.LAYER_COUNT, # LAYER_COUNT: 9
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT, # cfg.MODEL.MAX_CHANNEL_COUNT : 512
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
        truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF, # _C.MODEL.TRUNCATIOM_CUTOFF = 8
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=3)

model_dict = {
        'generator_s': model.generator,
        'mapping_fl_s': model.mapping,
        'dlatent_avg': model.dlatent_avg,  # dlatent_avg 平均人脸的位置
    }
#print('model.dlatent_avg_1:'+str(model.dlatent_avg.buff.data)) #[18,512],中心变量, 默认为0


logger = logging.getLogger("logger")
checkpointer = Checkpointer(cfg, model_dict, logger=logger, save=True)
checkpointer.load(file_name='./pre-model/karras2019stylegan-bedrooms-256x256.pth') #读取模型

Gm = model.mapping
Gs = model.generator
tensor = model.dlatent_avg
#-----------原版：前8层潜变量裁剪比例0.7后向中心变量拉近----------

# set_seed(5) #随机数测试，及Gmap测试
# latents = np.random.randn(1, 512)
# latents = torch.tensor(latents).float()
# latents = Gm(latents) #[1,512] -> [1,18,512]

# layer_idx = torch.arange(18)[np.newaxis, :, np.newaxis] # shape:[1,18,1], layer_idx = [0,1,2,3,4,5,6]
# ones = torch.ones(layer_idx.shape, dtype=torch.float32) # shape:[1,18,1], ones = [1,1,1,1,1,1,1,1]
# coefs = torch.where(layer_idx < 8, 0.7 * ones, ones) 
# style = torch.lerp(model.dlatent_avg.buff.data, latents, coefs)
#style_avg = model.dlatent_avg.buff.data.view(1,18,512)
#print(model.dlatent_avg.buff.data.shape) # (18,512) , 中心向量
#center_tensor = model.dlatent_avg.buff.data
#torch.save(center_tensor, 'center_tensor.pt' )
#center_tensor2 = torch.load('center_tensor.pt')


# print(style_avg.shape)
# img_avg, img = Gs1.forward(style_avg,8), Gs1.forward(style,8)
# save_image((img_avg+1)/2, 'dlatent_avg.png') #中心向量
# save_image((img+1)/2, 'sample_avg.png') #中心向量

#----------------单独存储Gs/Gm------------ 
Gm1 = model.mapping # 对象转存为dict
torch.save(Gm1.state_dict(),'./bedrooms256_Gm_dict.pth')

#Gm2 = Mapping(num_layers=16, mapping_layers=8, latent_size=512, dlatent_size=512, mapping_fmaps=512)
#Gm2 = torch.load('./cat512Gm.pth')
#torch.save(Gm2.state_dict(), './cat512_Gm_dict.pth')

Gs = model.generator
torch.save(Gs.state_dict(),'./bedrooms256_Gs_dict.pth')

center_tensor = model.dlatent_avg.buff.data
print(center_tensor.shape)
torch.save(center_tensor,'./bedrooms256_tensor.pt')

#-----------------测试Gm_dict--------------
# G = Generator(startf=16, maxf=512, layer_count=9, latent_size=512, channels=3)
# G.load_state_dict(torch.load('./Gs_dict.pth'))
# Gm = Mapping(num_layers=18, mapping_layers=8, latent_size=512, dlatent_size=512, mapping_fmaps=512)
# Gm.load_state_dict(torch.load('./Gm_dict.pth')) 

# for i in range(10,15):
# #i = 0
#     set_seed(i)
#     latents = torch.randn(10, 512)
#     print(latents[0,:20])
#     latents = Gm(latents) 
#     print(latents[0,1,:20])
# #init.ones_(G.const) #改变G训练好的const，会导致生成失败
# #print(G.const)
#     img = G.forward(latents,8)
#     save_image((img+1)/2, 'seed%d.png'%i)
