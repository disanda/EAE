import numpy as np
import torch
from configs.defaults import get_cfg_defaults
import argparse
from module.model import Model
from module.net import Generator
from nativeUtils.checkpointer import Checkpointer
import logging
import torchvision
from torchvision.utils import save_image
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="StyleGAN")
parser.add_argument(
        "--config-file",
        #default="configs/experiment_ffhq.yaml",
        default="configs/cat-bedroom-256.yaml",
        metavar="FILE",
        type=str,) # args.config_file
args = parser.parse_args()
cfg = get_cfg_defaults()
cfg.merge_from_file(args.config_file)
cfg.freeze()

model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count= cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
        truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=3)
model.eval()

model_dict = {
        'generator_s': model.generator,
        'mapping_fl_s': model.mapping,
        'dlatent_avg': model.dlatent_avg,
    }

logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)
print(cfg)
checkpointer = Checkpointer(cfg, model_dict, logger=logger, save=True)
#checkpointer.load(file_name='./pre-model/karras2019stylegan-ffhq.pth')
checkpointer.load(file_name='/Users/apple/Desktop/myStyle/model-result-v1/pre-model/karras2019stylegan-cats-256x256.pth')

#-----------random-w----------------- 在同seed下随机生成图像和官方pre-trained一致
rnd = np.random.RandomState(98)
latents = rnd.randn(1, cfg.MODEL.LATENT_SPACE_SIZE)
sample = torch.tensor(latents).float()

with torch.no_grad():
    save_image((model.generate(lod=6, blend_factor=1, z=sample, remove_blob= False)+1)/2, 'sample-cat-2.png') # model.generate()输入z:[-1,512], 经过Gmap后处理为[-1,18,512] 再经过Gs
#lod=6 -> 256 / lod=7 -> 512 / lod=8 -> 1024
#------------------trump w+------------ 用川普的潜码编辑
# donald_trump = np.load('./direction/donald_trump_01.npy') #[18, 512]
# i=5
# #donald_trump[i,:] = 0
# #print(donald_trump)
# donald_trump = donald_trump.reshape(1,18,512)
# donald_trump=torch.tensor(donald_trump)

# images=model.generator.forward(donald_trump,8) # lod=8,即1024,  和model.generate()不同,model.generator.forward()输入18层潜变量
# images=(images + 1)/2
# save_image(images,'trump.png')

#-----------------direnction with w+------- 用川普和属性的潜码编辑
# donald_trump = np.load('./direction/donald_trump_01.npy') #[18, 512]
# donald_trump = torch.tensor(donald_trump).float()
# direction = np.load('./direction/stylegan_ffhq_gender_w_boundary.npy')#[18, 512] 
# direction = torch.tensor(direction).float()
# direction_2 = direction.expand(18,512) # [1, 512] interfaceGAN
# print(direction_2.shape)

# i=-1 #属性向量加成
# j=3 #Latent code 层数: j+1
# layers=1
# seq = donald_trump
# seq[j:j+layers] = (seq+i*direction_2)[j:j+layers] # 选择第i-j层的潜码
# #seq = seq+i*direction_2
# seq = seq.reshape(1,18,512)
# with torch.no_grad():
# 	img = model.generator.forward(seq,8)

# save_image((img+1)/2,'smile-layer-%d-alllyer-%d-i-%f.png'%(j+1,layers,i))
# print('done')

#----------------------EAE Editing-------------
# from PIL import Image
# import module.BE_v2 as BE
# E = BE.BE()
# E.load_state_dict(torch.load('./pre-model/E_model_ep10000.pth',map_location=torch.device('cpu')))
# #Gs = Generator(startf=16, maxf=512, layer_count=9, latent_size=512, channels=3)
# #Gs.load_state_dict(torch.load('./pre-model/Gs_dict.pth',map_location=torch.device('cpu')))

# loader = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# def image_loader(image_name):
# 	image = Image.open(image_name).convert('RGB')
# 	#image = image.resize((1024,1024))
# 	image = loader(image).unsqueeze(0)
# 	return image.to(torch.float)

# imgs1=image_loader('./msk_align.png')
# imgs1 = imgs1*2-1

# with torch.no_grad():
# 	const1,w1 = E(imgs1)
# 	#imgs2 = Gs.forward(w1,8)
# w = w1[0]

# direction = np.load('./direction/stylegan_ffhq_eyeglasses_w_boundary.npy') #[[1, 512] interfaceGAN
# direction = torch.tensor(direction).float()
# direction_2 = direction.expand(18,512) 
# # direction_2 = np.load('./direction/smile.npy')#[18, 512] 
# # direction_2 = torch.tensor(direction_2).float()
# i=350 #属性向量加成
# j=0 #Latent code 层数: j+1
# layers=-1+5
# w[j:j+layers] = (w+i*direction_2)[j:j+layers] # 选择第i-j层的潜码
# #seq = seq+i*direction_2
# w = w.reshape(1,18,512)
# with torch.no_grad():
# 	#img2 = Gs.forward(seq,8)
# 	img2 = model.generator.forward(w1,8)
# torchvision.utils.save_image(img2*0.5+0.5, './msk_img_glasses_i%d_j%d_layers%d.png'%(i,j,layers))
