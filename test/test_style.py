from PIL import Image
import torch
import torchvision

loader = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

def image_loader(image_name):
	image = Image.open(image_name).convert('RGB')
	image = image.resize((256,256))
	image = loader(image).unsqueeze(0)
	return image.to(torch.float)

img1=image_loader('./wwm_align_2.png')
img1 = img1*2-1 # [0,1] -> [-1,1]

img2=image_loader('./style.png')
img2 = img2*2-1 

print(img1.shape)
print(img2.shape)

print(img2.mean(dim=(2,3),keepdim=True).shape) #求H,W的mean
style_mean1 = img1.mean(dim=(2,3),keepdim=True)
style_std1 = img1.std(dim=(2, 3), keepdim=True)

img1_s = (img1 - style_mean1)/style_std1

style_mean2 = img2.mean(dim=(2,3),keepdim=True)
style_std2 = img2.std(dim=(2, 3), keepdim=True)

img2_s = (img2 - style_mean2)/style_std2

imgT=img1+img2_s+img1_s


torchvision.utils.save_image(imgT*0.5+0.5,'wwn_T6.png')
