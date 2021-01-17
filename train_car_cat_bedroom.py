import os
import torch
import torchvision
from module.net import * # Generator,Mapping
import module.BE_v2 as BE
from module.custom_adam import LREQAdam
import lpips
from torch.nn import functional as F
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def set_seed(seed): #随机数设置
	np.random.seed(seed)
	#random.seed(seed)
	torch.manual_seed(seed) # cpu
	torch.cuda.manual_seed_all(seed)  # gpu
	torch.backends.cudnn.deterministic = True

def train(avg_tensor = None, coefs=0):
	Gs = Generator(startf=64, maxf=512, layer_count=7, latent_size=512, channels=3) # 32->512 layer_count=8 / 64->256 layer_count=7
	Gs.load_state_dict(torch.load('./pre-model/cat/cat256_Gs_dict.pth'))
	Gm = Mapping(num_layers=14, mapping_layers=8, latent_size=512, dlatent_size=512, mapping_fmaps=512) #num_layers: 14->256 / 16->512 / 18->1024
	Gm.load_state_dict(torch.load('./pre-model/cat/cat256_Gm_dict.pth')) 
	Gm.buffer1 = avg_tensor
	E = BE.BE(startf=64, maxf=512, layer_count=7, latent_size=512, channels=3)
	#E.load_state_dict(torch.load('/_yucheng/myStyle/EAE/result/EB_cars_v1/models/E_model_ep135000.pth'))
	Gs.cuda()
	#Gm.cuda()
	E.cuda()
	const_ = Gs.const

	E_optimizer = LREQAdam([{'params': E.parameters()},], lr=0.0015, betas=(0.0, 0.99), weight_decay=0)

	loss_all=0
	loss_mse = torch.nn.MSELoss()
	loss_lpips = lpips.LPIPS(net='vgg').to('cuda')
	loss_kl = torch.nn.KLDivLoss()

	batch_size = 10
	const1 = const_.repeat(batch_size,1,1,1)
	for epoch in range(0,250001):
		set_seed(epoch%30000)
		latents = torch.randn(batch_size, 512) #[32, 512]
		with torch.no_grad(): #这里需要生成图片和变量
			w1 = Gm(latents,coefs_m=coefs).to('cuda') #[batch_size,18,512]
			imgs1 = Gs.forward(w1,6) # 7->512 / 6->256

		const2,w2 = E(imgs1.cuda())

		imgs2=Gs.forward(w2,6)

		E_optimizer.zero_grad()
#loss1
		#loss_img_mse = loss_mse(imgs1,imgs2)
		loss_img_mse_c1 = loss_mse(imgs1[:,0],imgs2[:,0])
		loss_img_mse_c2 = loss_mse(imgs1[:,1],imgs2[:,1])
		loss_img_mse_c3 = loss_mse(imgs1[:,2],imgs2[:,2])
		loss_img_mse = max(loss_img_mse_c1,loss_img_mse_c2,loss_img_mse_c3)

		loss_img_lpips = loss_lpips(imgs1,imgs2).mean()

		y1_imgs, y2_imgs = torch.nn.functional.softmax(imgs1),torch.nn.functional.softmax(imgs2)
		loss_kl_img = loss_kl(torch.log(y2_imgs),y1_imgs) #D_kl(True=y1_imgs||Fake=y2_imgs)
		loss_kl_img = torch.where(torch.isnan(loss_kl_img),torch.full_like(loss_kl_img,0), loss_kl_img)
		loss_kl_img = torch.where(torch.isinf(loss_kl_img),torch.full_like(loss_kl_img,1), loss_kl_img)

		loss_1 = 5*loss_img_mse + 3*loss_img_lpips + loss_kl_img
		loss_1.backward(retain_graph=True)
		E_optimizer.step()
#loss2
		imgs_column1 = imgs1[:,:,imgs1.shape[2]//5:-imgs1.shape[2]//5,:] # w,h
		imgs_column2 = imgs2[:,:,imgs2.shape[2]//5:-imgs2.shape[2]//5,:]
		loss_img_mse_column = loss_mse(imgs_column1,imgs_column2)
		loss_img_lpips_column = loss_lpips(imgs_column1,imgs_column2).mean()

		loss_2 = 9*loss_img_mse_column +5*loss_img_lpips_column
		loss_2.backward(retain_graph=True)
		E_optimizer.step()
#loss3
		imgs_center1 = imgs1[:,:,imgs1.shape[2]//10:-imgs1.shape[2]//10,imgs1.shape[3]//10:-imgs1.shape[3]//10]
		imgs_center2 = imgs2[:,:,imgs2.shape[2]//10:-imgs2.shape[2]//10,imgs2.shape[3]//10:-imgs2.shape[3]//10]
		loss_img_mse_center = loss_mse(imgs_center1,imgs_center2)
		loss_img_lpips_center = loss_lpips(imgs_center1,imgs_center2).mean()

		# imgs_blob1 = imgs1[:,:,924:,924:]
		# imgs_blob2 = imgs2[:,:,924:,924:]
		# loss_img_mse_blob = loss_mse(imgs_blob1,imgs_blob2)

		loss_3 = 13*loss_img_mse_center +7*loss_img_lpips_center #+ loss_img_mse_blob
		loss_3.backward(retain_graph=True)
		#loss_x = loss_1+loss_2+loss_3
		#loss_x.backward(retain_graph=True)
		E_optimizer.step()
#loss4
		loss_c = loss_mse(const1,const2) #没有这个const，梯度起初没法快速下降，很可能无法收敛, 这个惩罚即乘0.1后,效果大幅提升！
		loss_c_m = loss_mse(const1.mean(),const2.mean())
		loss_c_s = loss_mse(const1.std(),const2.std())

		loss_w = loss_mse(w1,w2)
		loss_w_m = loss_mse(w1.mean(),w2.mean()) #初期一会很大10,一会很小0.0001
		loss_w_s = loss_mse(w1.std(),w2.std()) #后期一会很大，一会很小

		y1, y2 = torch.nn.functional.softmax(const1),torch.nn.functional.softmax(const2)
		loss_kl_c = loss_kl(torch.log(y2),y1)
		loss_kl_c = torch.where(torch.isnan(loss_kl_c),torch.full_like(loss_kl_c,0), loss_kl_c)
		loss_kl_c = torch.where(torch.isinf(loss_kl_c),torch.full_like(loss_kl_c,1), loss_kl_c)

		w1_kl, w2_kl = torch.nn.functional.softmax(w1),torch.nn.functional.softmax(w2)
		loss_kl_w = loss_kl(torch.log(w2_kl),w1_kl) #D_kl(True=y1_imgs||Fake=y2_imgs)
		loss_kl_w = torch.where(torch.isnan(loss_kl_w),torch.full_like(loss_kl_w,0), loss_kl_w)
		loss_kl_w = torch.where(torch.isinf(loss_kl_w),torch.full_like(loss_kl_w,1), loss_kl_w)

		loss_4 = 0.02*loss_c+0.03*loss_c_m+0.03*loss_c_s+0.02*loss_w+0.03*loss_w_m+0.03*loss_w_s+ loss_kl_w  + loss_kl_c
		loss_4.backward(retain_graph=True)
		E_optimizer.step()

		loss_all =  loss_1  + loss_4 + loss_2 + loss_3
		print('i_'+str(epoch)+'--loss_all__:'+str(loss_all.item())+'--loss_mse:'+str(loss_img_mse.item())+'--loss_lpips:'+str(loss_img_lpips.item())+'--loss_kl_img:'+str(loss_kl_img.item()))
		print('loss_img_mse_column:'+str(loss_img_mse_column.item())+'loss_img_lpips_column:'+str(loss_img_lpips_column.item())\
			+'--loss_img_mse_center:'+str(loss_img_mse_center.item())+'--loss_lpips_center:'+str(loss_img_lpips_center.item()))
		print('loss_w:'+str(loss_w.item())+'--loss_w_m:'+str(loss_w_m.item())+'--loss_w_s:'+str(loss_w_s.item())+'--loss_kl_w:'+str(loss_kl_w.item())\
			+'--loss_c:'+str(loss_c.item())+'--loss_c_m:'+str(loss_c_m.item())+'--loss_c_s:'+str(loss_c_s.item())+'--loss_kl_c:'+str(loss_kl_c.item()))
		print('-')

		if epoch % 100 == 0:
			n_row = 6
			test_img = torch.cat((imgs1[:n_row],imgs2[:n_row]))*0.5+0.5
			torchvision.utils.save_image(test_img, resultPath1_1+'/ep%d.jpg'%(epoch),nrow=n_row) # nrow=3
			with open(resultPath+'/Loss.txt', 'a+') as f:
				print('i_'+str(epoch)+'--loss_all__:'+str(loss_all.item())+'--loss_mse:'+str(loss_img_mse.item())+'--loss_lpips:'+str(loss_img_lpips.item())+'--loss_kl_img:'+str(loss_kl_img.item()),file=f)
				print('loss_img_mse_column:'+str(loss_img_mse_column.item())+'loss_img_lpips_column:'+str(loss_img_lpips_column.item())\
				+'--loss_img_mse_center:'+str(loss_img_mse_center.item())+'--loss_lpips_center:'+str(loss_img_lpips_center.item()),file=f)
				print('loss_w:'+str(loss_w.item())+'--loss_w_m:'+str(loss_w_m.item())+'--loss_w_s:'+str(loss_w_s.item())+'--loss_kl_w:'+str(loss_kl_w.item())+'--loss_c:'+str(loss_c.item())\
				+'--loss_c_m:'+str(loss_c_m.item())+'--loss_c_s:'+str(loss_c_s.item())+'--loss_kl_c:'+str(loss_kl_c.item()),file=f)
			if epoch % 5000 == 0:
				torch.save(E.state_dict(), resultPath1_2+'/E_model_ep%d.pth'%epoch)
				#torch.save(Gm.buffer1,resultPath1_2+'/center_tensor_ep%d.pt'%epoch)

if __name__ == "__main__":
	resultPath = "./result/EB_cat_v1"
	if not os.path.exists(resultPath): os.mkdir(resultPath)

	resultPath1_1 = resultPath+"/imgs"
	if not os.path.exists(resultPath1_1): os.mkdir(resultPath1_1)

	resultPath1_2 = resultPath+"/models"
	if not os.path.exists(resultPath1_2): os.mkdir(resultPath1_2)

	center_tensor = torch.load('./pre-model/cat/cat256-center_tensor.pt')
	layer_num = 14 # 14->256 / 16 -> 512  / 18->1024 
	layer_idx = torch.arange(layer_num)[np.newaxis, :, np.newaxis] # shape:[1,18,1], layer_idx = [0,1,2,3,4,5,6。。。，17]
	ones = torch.ones(layer_idx.shape, dtype=torch.float32) # shape:[1,18,1], ones = [1,1,1,1,1,1,1,1]
	coefs_ = torch.where(layer_idx < 8, 0.7 * ones, ones) # 18个变量前8个裁剪比例truncation_psi [0.7,0.7,...,1,1,1] 

	train(avg_tensor=center_tensor,coefs=coefs_)




