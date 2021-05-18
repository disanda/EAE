#更清楚的比较分为:
#image space: MSE -> SSIM  -> Lpips -> Gram-Cam  
#latent space: MSE -> Cosine Similarty -> Distribution Divergency
#在训练时加入write比较
import os
from skimage import io
import cv2
import torch
import torchvision
from module.net import * # Generator,Mapping
import module.EAE_model.BE_v2 as BE
from module.custom_adam import LREQAdam
import lpips
from torch.nn import functional as F
import metric.pytorch_ssim as pytorch_ssim
from metric.grad_cam import GradCAM, GradCamPlusPlus, GuidedBackPropagation, mask2cam
import tensorboardX
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def set_seed(seed): #随机数设置
    np.random.seed(seed)
    #random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True

def train(avg_tensor = None, coefs=0, tensor_writer=None):
    Gs = Generator(startf=64, maxf=512, layer_count=7, latent_size=512, channels=3) # 32->512 layer_count=8 / 64->256 layer_count=7
    Gs.load_state_dict(torch.load('./pre-model/cat/cat256_Gs_dict.pth'))
    Gm = Mapping(num_layers=14, mapping_layers=8, latent_size=512, dlatent_size=512, mapping_fmaps=512) #num_layers: 14->256 / 16->512 / 18->1024
    Gm.load_state_dict(torch.load('./pre-model/cat/cat256_Gm_dict.pth')) 
    Gm.buffer1 = avg_tensor
    E = BE.BE(startf=64, maxf=512, layer_count=7, latent_size=512, channels=3)
    E.load_state_dict(torch.load('/_yucheng/myStyle/myStyle-v1/EAE-car-cat/result/EB_cat_cosine_v2/E_model_ep80000.pth'))
    Gs.cuda()
    #Gm.cuda()
    E.cuda()
    const_ = Gs.const
    writer = tensor_writer

    E_optimizer = LREQAdam([{'params': E.parameters()},], lr=0.0015, betas=(0.0, 0.99), weight_decay=0)
    loss_mse = torch.nn.MSELoss()
    loss_lpips = lpips.LPIPS(net='vgg').to('cuda')
    loss_kl = torch.nn.KLDivLoss()
    ssim_loss = pytorch_ssim.SSIM()

    batch_size = 3
    const1 = const_.repeat(batch_size,1,1,1)

    vgg16 = torchvision.models.vgg16(pretrained=True).cuda()
    final_layer = None
    for name, m in vgg16.named_modules():
        if isinstance(m, nn.Conv2d):
            final_layer = name
    grad_cam_plus_plus = GradCamPlusPlus(vgg16, final_layer)
    gbp = GuidedBackPropagation(vgg16)


    it_d = 0
    for epoch in range(0,250001):
        set_seed(epoch%30000)
        latents = torch.randn(batch_size, 512) #[32, 512]
        with torch.no_grad(): #这里需要生成图片和变量
            w1 = Gm(latents,coefs_m=coefs).to('cuda') #[batch_size,18,512]
            imgs1 = Gs.forward(w1,6) # 7->512 / 6->256

        const2,w2 = E(imgs1.cuda())

        imgs2=Gs.forward(w2,6)

        E_optimizer.zero_grad()

#Image Space
        mask_1 = grad_cam_plus_plus(imgs1,None) #[c,1,h,w]
        mask_2 = grad_cam_plus_plus(imgs2,None)
        #imgs1.retain_grad()
        #imgs2.retain_grad()
        imgs1_ = imgs1.detach().clone()
        imgs1_.requires_grad = True
        imgs2_ = imgs2.detach().clone()
        imgs2_.requires_grad = True
        grad1 = gbp(imgs1_) # [n,c,h,w]
        grad2 = gbp(imgs2_)

#Mask_Cam
        mask_1 = mask_1.cuda().float()
        mask_1.requires_grad=True
        mask_2 = mask_2.cuda().float()
        mask_2.requires_grad=True
        loss_mask_mse_1 = loss_mse(mask_1,mask_2)
        loss_mask_mse_2 = loss_mse(mask_1.mean(),mask_2.mean())
        loss_mask_mse_3 = loss_mse(mask_1.std(),mask_2.std())
        loss_mask_mse = loss_mask_mse_1 + loss_mask_mse_2 + loss_mask_mse_3

        ssim_value = pytorch_ssim.ssim(mask_1, mask_2) # while ssim_value<0.999:
        loss_mask_ssim = 1-ssim_loss(mask_1, mask_2)

        loss_mask_lpips = loss_lpips(mask_1,mask_2).mean()

        mask1_kl, mask2_kl = torch.nn.functional.softmax(mask_1),torch.nn.functional.softmax(mask_2)
        loss_kl_mask = loss_kl(torch.log(mask2_kl),mask1_kl) #D_kl(True=y1_imgs||Fake=y2_imgs)
        loss_kl_mask = torch.where(torch.isnan(loss_kl_w),torch.full_like(loss_kl_w,0), loss_kl_w)
        loss_kl_mask = torch.where(torch.isinf(loss_kl_w),torch.full_like(loss_kl_w,1), loss_kl_w)

        mask1_cos = mask1.view(-1)
        mask2_cos = mask2.view(-1)
        loss_cosine_w = 1 - mask1_cos.dot(mask2_cos)/(torch.sqrt(mask1_cos.dot(mask1_cos))*torch.sqrt(mask2_cos.dot(mask2_cos))) #[-1,1],-1:反向相反，1:方向相同

        loss_mask = loss_mask_mse + loss_mask_ssim + loss_mask_lpips + loss_kl_mask + loss_cosine_w
        E_optimizer.zero_grad()
        loss_mask.backward(retain_graph=True)
        E_optimizer.step()

#Grad
        grad1 = grad1.cuda().float()
        grad1.requires_grad=True
        grad2 = grad2.cuda().float()
        grad2.requires_grad=True
        loss_grad_mse = loss_mse(grad1,grad2)
        E_optimizer.zero_grad()
        loss_grad_mse.backward(retain_graph=True)
        E_optimizer.step()

        ssim_value = pytorch_ssim.ssim(grad1, grad2) # while ssim_value<0.999:
        loss_grad_ssim = 1-ssim_loss(grad1, grad2)
        E_optimizer.zero_grad()
        loss_grad_ssim.backward(retain_graph=True)
        E_optimizer.step()

        loss_grad_lpips = loss_lpips(grad1,grad2).mean()
        E_optimizer.zero_grad()
        loss_grad_lpips.backward(retain_graph=True)
        E_optimizer.step()

        grad1 = grad1.cuda().float()
        grad1.requires_grad=True
        grad2 = grad2.cuda().float()
        grad2.requires_grad=True
        loss_grad_mse_1 = loss_mse(grad1,grad2)
        loss_grad_mse_2 = loss_mse(grad1.mean(),grad2.mean())
        loss_grad_mse_3 = loss_mse(mask_1.std(),mask_2.std())
        loss_mask_mse = loss_mask_mse_1 + loss_mask_mse_2 + loss_mask_mse_3

        ssim_value = pytorch_ssim.ssim(mask_1, mask_2) # while ssim_value<0.999:
        loss_mask_ssim = 1-ssim_loss(mask_1, mask_2)

        loss_mask_lpips = loss_lpips(mask_1,mask_2).mean()

        mask1_kl, mask2_kl = torch.nn.functional.softmax(mask_1),torch.nn.functional.softmax(mask_2)
        loss_kl_mask = loss_kl(torch.log(mask2_kl),mask1_kl) #D_kl(True=y1_imgs||Fake=y2_imgs)
        loss_kl_mask = torch.where(torch.isnan(loss_kl_w),torch.full_like(loss_kl_w,0), loss_kl_w)
        loss_kl_mask = torch.where(torch.isinf(loss_kl_w),torch.full_like(loss_kl_w,1), loss_kl_w)

        mask1_cos = mask1.view(-1)
        mask2_cos = mask2.view(-1)
        loss_cosine_w = 1 - mask1_cos.dot(mask2_cos)/(torch.sqrt(mask1_cos.dot(mask1_cos))*torch.sqrt(mask2_cos.dot(mask2_cos))) #[-1,1],-1:反向相反，1:方向相同

        loss_mask = loss_mask_mse + loss_mask_ssim + loss_mask_lpips + loss_kl_mask + loss_cosine_w
        E_optimizer.zero_grad()
        loss_mask.backward(retain_graph=True)
        E_optimizer.step()



#Image
        loss_img_mse = loss_mse(imgs1,imgs2)
        E_optimizer.zero_grad()
        loss_img_mse.backward(retain_graph=True)
        E_optimizer.step()

        ssim_value = pytorch_ssim.ssim(imgs1, imgs2) # while ssim_value<0.999:
        loss_img_ssim = 1-ssim_loss(imgs1, imgs2)
        E_optimizer.zero_grad()
        loss_img_ssim.backward(retain_graph=True)
        E_optimizer.step()

        loss_img_lpips = loss_lpips(imgs1,imgs2).mean()
        E_optimizer.zero_grad()
        loss_img_lpips.backward(retain_graph=True)
        E_optimizer.step()


#Latent Space
# W
        loss_w = loss_mse(w1,w2)
        loss_w_m = loss_mse(w1.mean(),w2.mean()) #初期一会很大10,一会很小0.0001
        loss_w_s = loss_mse(w1.std(),w2.std()) #后期一会很大，一会很小

        w1_kl, w2_kl = torch.nn.functional.softmax(w1),torch.nn.functional.softmax(w2)
        loss_kl_w = loss_kl(torch.log(w2_kl),w1_kl) #D_kl(True=y1_imgs||Fake=y2_imgs)
        loss_kl_w = torch.where(torch.isnan(loss_kl_w),torch.full_like(loss_kl_w,0), loss_kl_w)
        loss_kl_w = torch.where(torch.isinf(loss_kl_w),torch.full_like(loss_kl_w,1), loss_kl_w)

        w1_cos = w1.view(-1)
        w2_cos = w2.view(-1)
        loss_cosine_w = 1 - w1_cos.dot(w2_cos)/(torch.sqrt(w1_cos.dot(w1_cos))*torch.sqrt(w2_cos.dot(w2_cos))) #[-1,1],-1:反向相反，1:方向相同
# C
        loss_c = loss_mse(const1,const2) #没有这个const，梯度起初没法快速下降，很可能无法收敛, 这个惩罚即乘0.1后,效果大幅提升！
        loss_c_m = loss_mse(const1.mean(),const2.mean())
        loss_c_s = loss_mse(const1.std(),const2.std())

        y1, y2 = torch.nn.functional.softmax(const1),torch.nn.functional.softmax(const2)
        loss_kl_c = loss_kl(torch.log(y2),y1)
        loss_kl_c = torch.where(torch.isnan(loss_kl_c),torch.full_like(loss_kl_c,0), loss_kl_c)
        loss_kl_c = torch.where(torch.isinf(loss_kl_c),torch.full_like(loss_kl_c,1), loss_kl_c)

        c_cos1 = const1.view(-1)
        c_cos2 = const2.view(-1)
        loss_cosine_c = 1 - c_cos1.dot(c_cos2)/(torch.sqrt(c_cos1.dot(c_cos1))*torch.sqrt(c_cos2.dot(c_cos2))) 


        loss_ls_all = loss_w+loss_w_m+loss_w_s+loss_kl_w+loss_cosine_w+\
                        loss_c+loss_c_m+loss_c_s+loss_kl_c+loss_cosine_c
        loss_ls_all.backward(retain_graph=True)
        E_optimizer.step()

        print('i_'+str(epoch))
        print('---------ImageSpace--------')
        print('loss_mask_mse:'+str(loss_mask_mse.item())+'--loss_mask_ssim:'+str(loss_mask_ssim.item())+'--loss_mask_lpips:'+str(loss_mask_lpips.item()))
        print('loss_grad_mse:'+str(loss_grad_mse.item())+'--loss_grad_ssim:'+str(loss_grad_ssim.item())+'--loss_grad_lpips:'+str(loss_grad_lpips.item()))
        print('loss_img_mse:'+str(loss_img_mse.item())+'--loss_img_ssim:'+str(loss_img_ssim.item())+'--loss_img_lpips:'+str(loss_img_lpips.item()))
        print('---------LatentSpace--------')
        print('loss_w:'+str(loss_w.item())+'--loss_w_m:'+str(loss_w_m.item())+'--loss_w_s:'+str(loss_w_s.item()))
        print('loss_kl_w:'+str(loss_kl_w.item())+'--loss_cosine_w:'+str(loss_cosine_w.item()))
        print('loss_c:'+str(loss_c.item())+'--loss_c_m:'+str(loss_c_m.item())+'--loss_c_s:'+str(loss_c_s.item()))
        print('loss_kl_c:'+str(loss_kl_c.item())+'--loss_cosine_c:'+str(loss_cosine_c.item()))

        it_d += 1
        writer.add_scalar('loss_mask_mse', loss_mask_mse, global_step=it_d)
        writer.add_scalar('loss_mask_ssim', loss_mask_ssim, global_step=it_d)
        writer.add_scalar('loss_mask_lpips', loss_mask_lpips, global_step=it_d)
        writer.add_scalar('loss_grad_mse',  loss_grad_mse, global_step=it_d)
        writer.add_scalar('loss_grad_ssim', loss_grad_ssim, global_step=it_d)
        writer.add_scalar('loss_grad_lpips', loss_grad_lpips, global_step=it_d)
        writer.add_scalar('loss_img_mse', loss_img_mse, global_step=it_d)
        writer.add_scalar('loss_img_ssim', loss_img_ssim, global_step=it_d)
        writer.add_scalar('loss_img_lpips', loss_img_lpips, global_step=it_d)
        writer.add_scalar('loss_w', loss_w, global_step=it_d)
        writer.add_scalar('loss_w_m', loss_w_m, global_step=it_d)
        writer.add_scalar('loss_w_s', loss_w_s, global_step=it_d)
        writer.add_scalar('loss_kl_w',  loss_kl_w, global_step=it_d)
        writer.add_scalar('loss_cosine_w', loss_cosine_w, global_step=it_d)
        writer.add_scalar('loss_c', loss_c, global_step=it_d)
        writer.add_scalar('loss_c_m', loss_c_m, global_step=it_d)
        writer.add_scalar('loss_c_s', loss_c_s, global_step=it_d)
        writer.add_scalar('loss_kl_c',  loss_kl_c, global_step=it_d)
        writer.add_scalar('loss_cosine_c',  loss_cosine_c, global_step=it_d)
        writer.add_scalars('Image_Space', {'loss_mask_mse':loss_mask_mse,'loss_grad_mse':loss_grad_mse,'loss_img_mse':loss_img_mse}, global_step=it_d)
        writer.add_scalars('Image_Space', {'loss_mask_ssim':loss_mask_mse,'loss_grad_ssim':loss_grad_ssim,'loss_img_ssim':loss_img_ssim}, global_step=it_d)
        writer.add_scalars('Image_Space', {'loss_mask_lpips':loss_mask_lpips,'loss_grad_lpips':loss_grad_lpips,'loss_img_lpips':loss_img_lpips}, global_step=it_d)
        writer.add_scalars('Latent Space W', {'loss_w':loss_w,'loss_w_m':loss_w_m,'loss_w_s':loss_w_s,'loss_kl_w':loss_kl_w,'loss_cosine_w':loss_cosine_w}, global_step=it_d)
        writer.add_scalars('Latent Space C', {'loss_c':loss_c,'loss_c_m':loss_c_m,'loss_c_s':loss_c_s,'loss_kl_c':loss_kl_c,'loss_cosine_c':loss_cosine_c}, global_step=it_d)

        if epoch % 100 == 0:
            n_row = batch_size
            test_img = torch.cat((imgs1[:n_row],imgs2[:n_row]))*0.5+0.5
            torchvision.utils.save_image(test_img, resultPath1_1+'/ep%d.png'%(epoch),nrow=n_row) # nrow=3
            heatmap1,cam1 = mask2cam(mask_1,imgs1)
            heatmap2,cam2 = mask2cam(mask_2,imgs2)
            heatmap=torch.cat((heatmap1,heatmap1))
            cam=torch.cat((cam1,cam2))
            grads = torch.cat((grad1,grad2))
            grads = grads.data.cpu().numpy() # [n,c,h,w]
            grads -= np.max(np.min(grads), 0)
            grads /= np.max(grads)
            torchvision.utils.save_image(torch.tensor(heatmap),resultPath_grad_cam+'/heatmap_%d.png'%(epoch))
            torchvision.utils.save_image(torch.tensor(cam),resultPath_grad_cam+'/cam_%d.png'%(epoch))
            torchvision.utils.save_image(torch.tensor(grads),resultPath_grad_cam+'/gb_%d.png'%(epoch))
            with open(resultPath+'/Loss.txt', 'a+') as f:
                        print('i_'+str(epoch),file=f)
                        print('---------ImageSpace--------',file=f)
                        print('loss_mask_mse:'+str(loss_mask_mse.item())+'--loss_mask_ssim:'+str(loss_mask_ssim.item())+'--loss_mask_lpips:'+str(loss_mask_lpips.item()),file=f)
                        print('loss_grad_mse:'+str(loss_grad_mse.item())+'--loss_grad_ssim:'+str(loss_grad_ssim.item())+'--loss_grad_lpips:'+str(loss_grad_lpips.item()),file=f)
                        print('loss_img_mse:'+str(loss_img_mse.item())+'--loss_img_ssim:'+str(loss_img_ssim.item())+'--loss_img_lpips:'+str(loss_img_lpips.item()),file=f)
                        print('---------LatentSpace--------',file=f)
                        print('loss_w:'+str(loss_w.item())+'--loss_w_m:'+str(loss_w_m.item())+'--loss_w_s:'+str(loss_w_s.item()),file=f)
                        print('loss_kl_w:'+str(loss_kl_w.item())+'--loss_cosine_w:'+str(loss_cosine_w.item()),file=f)
                        print('loss_c:'+str(loss_c.item())+'--loss_c_m:'+str(loss_c_m.item())+'--loss_c_s:'+str(loss_c_s.item()),file=f)
                        print('loss_kl_c:'+str(loss_kl_c.item())+'--loss_cosine_c:'+str(loss_cosine_c.item()),file=f)
            if epoch % 5000 == 0:
                torch.save(E.state_dict(), resultPath1_2+'/E_model_ep%d.pth'%epoch)
                #torch.save(Gm.buffer1,resultPath1_2+'/center_tensor_ep%d.pt'%epoch)

if __name__ == "__main__":

    if not os.path.exists('./result'): os.mkdir('./result')

    resultPath = "./result/D2E_CAT_v2)"
    if not os.path.exists(resultPath): os.mkdir(resultPath)

    resultPath1_1 = resultPath+"/imgs"
    if not os.path.exists(resultPath1_1): os.mkdir(resultPath1_1)

    resultPath1_2 = resultPath+"/models"
    if not os.path.exists(resultPath1_2): os.mkdir(resultPath1_2)

    resultPath_grad_cam = resultPath+"/grad_cam"
    if not os.path.exists(resultPath_grad_cam): os.mkdir(resultPath_grad_cam)

    center_tensor = torch.load('./pre-model/cat/cat256-center_tensor.pt')
    layer_num = 14 # 14->256 / 16 -> 512  / 18->1024 
    layer_idx = torch.arange(layer_num)[np.newaxis, :, np.newaxis] # shape:[1,18,1], layer_idx = [0,1,2,3,4,5,6。。。，17]
    ones = torch.ones(layer_idx.shape, dtype=torch.float32) # shape:[1,18,1], ones = [1,1,1,1,1,1,1,1]
    coefs_ = torch.where(layer_idx < layer_num//2, 0.7 * ones, ones) # 18个变量前8个裁剪比例truncation_psi [0.7,0.7,...,1,1,1] 

    writer_path = os.path.join(resultPath, './summaries')
    if not os.path.exists(writer_path): os.mkdir(writer_path)
    writer = tensorboardX.SummaryWriter(writer_path)

    train(avg_tensor=center_tensor, coefs=coefs_, tensor_writer=writer)




