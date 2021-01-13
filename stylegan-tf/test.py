import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import os
import pickle
from torchvision.utils import save_image
import torch
import PIL.Image as Image
import tensorflow as tf

path = '../karras2019stylegan-ffhq-1024x1024.pkl'

tflib.init_tf()
with open(path, 'rb') as f:
	_G, _D, GS = pickle.load(f) #GS包含了Gm & Gs


#-----------random-w-----------------
# initial_dlatents = np.zeros((1, 18, 512)) # [-1,18,512]
# rnd = np.random.RandomState(5)
# latents = rnd.randn(1, 512)

# fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
# images = GS.run(latents, None, truncation_psi=0.7, randomize_noise=False, output_transform=fmt)
# images = images.squeeze(0) # [h,w,c] 
# print(images.shape)
# im = Image.fromarray(images)
# im.save('sample.png')

#------------------trump w+------------
donald_trump = np.load('ffhq_dataset/latent_representations/donald_trump_01.npy') #[18, 512]
donald_trump = donald_trump.reshape(1,18,512)
initial_dlatents = np.zeros((1, 18, 512)) # [-1,18,512]

images=GS.components.synthesis.run(donald_trump) #randomize_noise=False
images = tflib.convert_images_to_uint8(images) #[-1,1] -> [0,255]
images = images.eval().transpose(0,2,3,1) # tf.tensor -> np -> [-1,c,h,w] -> [-1,h,w,c]
print(images.shape)
im = Image.fromarray(images[0])
im.save('trump.png')