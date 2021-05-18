#更清楚的比较分为:
#image space: MSE -> SSIM  -> Lpips -> Gram-Cam  
#latent space: MSE -> Cosine Similarty -> Distribution Divergency
#在训练时加入write比较
##续：2021_May_18, 把损失函数统一放入函数
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

def space_loss(imgs1,imgs2,image_space=True,lpips_model=None):
    loss_mse = torch.nn.MSELoss()
    loss_kl = torch.nn.KLDivLoss()
    ssim_loss = pytorch_ssim.SSIM()
    loss_lpips = lpips_model

    loss_imgs_mse_1 = loss_mse(imgs1,imgs2)
    loss_imgs_mse_2 = loss_mse(imgs1.mean(),imgs2.mean())
    loss_imgs_mse_3 = loss_mse(imgs1.std(),imgs2.std())
    loss_imgs_mse = loss_imgs_mse_1 + loss_imgs_mse_2 + loss_imgs_mse_3

    imgs1_kl, imgs2_kl = torch.nn.functional.softmax(imgs1),torch.nn.functional.softmax(imgs2)
    loss_imgs_kl = loss_kl(torch.log(imgs2_kl),imgs1_kl) #D_kl(True=y1_imgs||Fake=y2_imgs)
    loss_imgs_kl = torch.where(torch.isnan(loss_imgs_kl),torch.full_like(loss_imgs_kl,0), loss_imgs_kl)
    loss_imgs_kl = torch.where(torch.isinf(loss_imgs_kl),torch.full_like(loss_imgs_kl,1), loss_imgs_kl)

    imgs1_cos = imgs1.view(-1)
    imgs2_cos = imgs2.view(-1)
    loss_imgs_cosine = 1 - imgs1_cos.dot(imgs2_cos)/(torch.sqrt(imgs1_cos.dot(imgs1_cos))*torch.sqrt(imgs2_cos.dot(imgs2_cos))) #[-1,1],-1:反向相反，1:方向相同

    if  image_space:
        ssim_value = pytorch_ssim.ssim(imgs1, imgs2) # while ssim_value<0.999:
        loss_imgs_ssim = 1-ssim_loss(imgs1, imgs2)
    else:
        loss_imgs_ssim = torch.tensor(0)

    if image_space:
        loss_imgs_lpips = loss_lpips(imgs1,imgs2).mean()
    else:
        loss_imgs_lpips = torch.tensor(0)

    loss_imgs = loss_imgs_mse + loss_imgs_kl + loss_imgs_cosine + loss_imgs_ssim + loss_imgs_lpips
    loss_info = [[loss_imgs_mse_1.item(),loss_imgs_mse_2.item(),loss_imgs_mse_3.item()], loss_imgs_kl.item(), loss_imgs_cosine.item(), loss_imgs_ssim.item(), loss_imgs_lpips.item()]
    return loss_imgs, loss_info

def train(avg_tensor = None, coefs=0, tensor_writer=None):
    Gs = Generator(startf=64, maxf=512, layer_count=7, latent_size=512, channels=3) # 32->512 layer_count=8 / 64->256 layer_count=7
    Gs.load_state_dict(torch.load('./pre-model/cat/cat256_Gs_dict.pth'))
    Gm = Mapping(num_layers=14, mapping_layers=8, latent_size=512, dlatent_size=512, mapping_fmaps=512) #num_layers: 14->256 / 16->512 / 18->1024
    Gm.load_state_dict(torch.load('./pre-model/cat/cat256_Gm_dict.pth')) 
    Gm.buffer1 = avg_tensor
    E = BE.BE(startf=64, maxf=512, layer_count=7, latent_size=512, channels=3)
    E.load_state_dict(torch.load('/_yucheng/myStyle/myStyle-v1/EAE-car-cat/pre-model/E_cat_v2_1_ep100000.pth'))
    Gs.cuda()
    #Gm.cuda()
    E.cuda()
    const_ = Gs.const
    writer = tensor_writer

    E_optimizer = LREQAdam([{'params': E.parameters()},], lr=0.0015, betas=(0.0, 0.99), weight_decay=0)
    loss_lpips = lpips.LPIPS(net='vgg').to('cuda')

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
        grad_1 = gbp(imgs1_) # [n,c,h,w]
        grad_2 = gbp(imgs2_)

    ##--Mask_Cam
        mask_1 = mask_1.cuda().float()
        mask_1.requires_grad=True
        mask_2 = mask_2.cuda().float()
        mask_2.requires_grad=True
        loss_mask, loss_mask_info = space_loss(mask_1,mask_2,lpips_model=loss_lpips)

        E_optimizer.zero_grad()
        loss_mask.backward(retain_graph=True)
        E_optimizer.step()

    ##--Grad
        grad_1 = grad_1.cuda().float()
        grad_1.requires_grad=True
        grad_2 = grad_2.cuda().float()
        grad_2.requires_grad=True
        loss_grad, loss_grad_info = space_loss(grad_1,grad_2,lpips_model=loss_lpips)

        E_optimizer.zero_grad()
        loss_grad.backward(retain_graph=True)
        E_optimizer.step()

    ##--Image
        loss_imgs, loss_imgs_info = space_loss(imgs1,imgs2,lpips_model=loss_lpips)
        E_optimizer.zero_grad()
        loss_imgs.backward(retain_graph=True)
        E_optimizer.step()

#Latent Space
    ##--W
        loss_w, loss_w_info = space_loss(w1,w2,image_space = False)
        E_optimizer.zero_grad()
        loss_w.backward(retain_graph=True)
        E_optimizer.step()

    ##--C
        loss_c, loss_c_info = space_loss(const1,const2,image_space = False)
        E_optimizer.zero_grad()
        loss_c.backward(retain_graph=True)
        E_optimizer.step()

        print('i_'+str(epoch))
        print('[loss_imgs_mse[img,img_mean,img_std], loss_imgs_ssim, loss_imgs_cosine, loss_kl_imgs, loss_imgs_lpips]')
        print('---------ImageSpace--------')
        print('loss_mask_info: %s'%loss_mask_info)
        print('loss_grad_info: %s'%loss_grad_info)
        print('loss_imgs_info: %s'%loss_imgs_info)
        print('---------LatentSpace--------')
        print('loss_w_info: %s'%loss_w_info)
        print('loss_c_info: %s'%loss_c_info)

        it_d += 1
        writer.add_scalar('loss_mask_mse', loss_mask_info[0][0], global_step=it_d)
        writer.add_scalar('loss_mask_mse_mean', loss_mask_info[0][1], global_step=it_d)
        writer.add_scalar('loss_mask_mse_std', loss_mask_info[0][2], global_step=it_d)
        writer.add_scalar('loss_mask_kl', loss_mask_info[1], global_step=it_d)
        writer.add_scalar('loss_mask_cosine', loss_mask_info[2], global_step=it_d)
        writer.add_scalar('loss_mask_ssim', loss_mask_info[3], global_step=it_d)
        writer.add_scalar('loss_mask_lpips', loss_mask_info[4], global_step=it_d)

        writer.add_scalar('loss_grad_mse', loss_grad_info[0][0], global_step=it_d)
        writer.add_scalar('loss_grad_mse_mean', loss_grad_info[0][1], global_step=it_d)
        writer.add_scalar('loss_grad_mse_std', loss_grad_info[0][2], global_step=it_d)
        writer.add_scalar('loss_grad_kl', loss_grad_info[1], global_step=it_d)
        writer.add_scalar('loss_grad_cosine', loss_grad_info[2], global_step=it_d)
        writer.add_scalar('loss_grad_ssim', loss_grad_info[3], global_step=it_d)
        writer.add_scalar('loss_grad_lpips', loss_grad_info[4], global_step=it_d)

        writer.add_scalar('loss_imgs_mse', loss_imgs_info[0][0], global_step=it_d)
        writer.add_scalar('loss_imgs_mse_mean', loss_imgs_info[0][1], global_step=it_d)
        writer.add_scalar('loss_imgs_mse_std', loss_imgs_info[0][2], global_step=it_d)
        writer.add_scalar('loss_imgs_kl', loss_imgs_info[1], global_step=it_d)
        writer.add_scalar('loss_imgs_cosine', loss_imgs_info[2], global_step=it_d)
        writer.add_scalar('loss_imgs_ssim', loss_imgs_info[3], global_step=it_d)
        writer.add_scalar('loss_imgs_lpips', loss_imgs_info[4], global_step=it_d)

        writer.add_scalar('loss_w_mse', loss_w_info[0][0], global_step=it_d)
        writer.add_scalar('loss_w_mse_mean', loss_w_info[0][1], global_step=it_d)
        writer.add_scalar('loss_w_mse_std', loss_w_info[0][2], global_step=it_d)
        writer.add_scalar('loss_w_kl', loss_w_info[1], global_step=it_d)
        writer.add_scalar('loss_w_cosine', loss_w_info[2], global_step=it_d)
        writer.add_scalar('loss_w_ssim', loss_w_info[3], global_step=it_d)
        writer.add_scalar('loss_w_lpips', loss_w_info[4], global_step=it_d)

        writer.add_scalar('loss_c_mse', loss_c_info[0][0], global_step=it_d)
        writer.add_scalar('loss_c_mse_mean', loss_c_info[0][1], global_step=it_d)
        writer.add_scalar('loss_c_mse_std', loss_c_info[0][2], global_step=it_d)
        writer.add_scalar('loss_c_kl', loss_c_info[1], global_step=it_d)
        writer.add_scalar('loss_c_cosine', loss_c_info[2], global_step=it_d)
        writer.add_scalar('loss_c_ssim', loss_c_info[3], global_step=it_d)
        writer.add_scalar('loss_c_lpips', loss_c_info[4], global_step=it_d)

        writer.add_scalars('Image_Space_MSE', {'loss_mask_mse':loss_imgs_info[0][0],'loss_grad_mse':loss_grad_info[0][0],'loss_img_mse':loss_imgs_info[0][0]}, global_step=it_d)
        writer.add_scalars('Image_Space_KL', {'loss_mask_kl':loss_mask_info[1],'loss_grad_cosine':loss_grad_info[1],'loss_imgs_cosine':loss_imgs_info[1]}, global_step=it_d)
        writer.add_scalars('Image_Space_Cosine', {'loss_mask_cosine':loss_mask_info[2],'loss_grad_cosine':loss_grad_info[2],'loss_imgs_cosine':loss_imgs_info[2]}, global_step=it_d)
        writer.add_scalars('Image_Space_SSIM', {'loss_mask_ssim':loss_mask_info[3],'loss_grad_ssim':loss_grad_info[3],'loss_img_ssim':loss_imgs_info[3]}, global_step=it_d)
        writer.add_scalars('Image_Space_Cosine', {'loss_mask_cosine':loss_mask_info[4],'loss_grad_cosine':loss_grad_info[4],'loss_imgs_cosine':loss_imgs_info[4]}, global_step=it_d)
        writer.add_scalars('Image_Space_Lpips', {'loss_mask_lpips':loss_mask_info[4],'loss_grad_lpips':loss_grad_info[4],'loss_img_lpips':loss_imgs_info[4]}, global_step=it_d)
        writer.add_scalars('Latent Space W', {'loss_w_mse':loss_w_info[0][0],'loss_w_mse_mean':loss_w_info[0][1],'loss_w_mse_std':loss_w_info[0][2],'loss_w_kl':loss_w_info[1],'loss_w_cosine':loss_w_info[2]}, global_step=it_d)
        writer.add_scalars('Latent Space C', {'loss_c_mse':loss_c_info[0][0],'loss_c_mse_mean':loss_w_info[0][1],'loss_c_mse_std':loss_w_info[0][2],'loss_c_kl':loss_w_info[1],'loss_c_cosine':loss_w_info[2]}, global_step=it_d)

        if epoch % 100 == 0:
            n_row = batch_size
            test_img = torch.cat((imgs1[:n_row],imgs2[:n_row]))*0.5+0.5
            torchvision.utils.save_image(test_img, resultPath1_1+'/ep%d.png'%(epoch),nrow=n_row) # nrow=3
            heatmap1,cam1 = mask2cam(mask_1,imgs1)
            heatmap2,cam2 = mask2cam(mask_2,imgs2)
            heatmap=torch.cat((heatmap1,heatmap1))
            cam=torch.cat((cam1,cam2))
            grads = torch.cat((grad_1,grad_2))
            grads = grads.data.cpu().numpy() # [n,c,h,w]
            grads -= np.max(np.min(grads), 0)
            grads /= np.max(grads)
            torchvision.utils.save_image(torch.tensor(heatmap),resultPath_grad_cam+'/heatmap_%d.png'%(epoch))
            torchvision.utils.save_image(torch.tensor(cam),resultPath_grad_cam+'/cam_%d.png'%(epoch))
            torchvision.utils.save_image(torch.tensor(grads),resultPath_grad_cam+'/gb_%d.png'%(epoch))
            with open(resultPath+'/Loss.txt', 'a+') as f:
                print('i_'+str(epoch),file=f)
                print('[loss_imgs_mse[img,img_mean,img_std], loss_imgs_kl, loss_imgs_cosine, loss_imgs_ssim, loss_imgs_lpips]',file=f)
                print('---------ImageSpace--------',file=f)
                print('loss_mask_info: %s'%loss_mask_info,file=f)
                print('loss_grad_info: %s'%loss_grad_info,file=f)
                print('loss_imgs_info: %s'%loss_imgs_info,file=f)
                print('---------LatentSpace--------',file=f)
                print('loss_w_info: %s'%loss_w_info,file=f)
                print('loss_c_info: %s'%loss_c_info,file=f)
            if epoch % 5000 == 0:
                torch.save(E.state_dict(), resultPath1_2+'/E_model_ep%d.pth'%epoch)
                #torch.save(Gm.buffer1,resultPath1_2+'/center_tensor_ep%d.pt'%epoch)

if __name__ == "__main__":

    if not os.path.exists('./result'): os.mkdir('./result')

    resultPath = "./result/D2E_CAT_v2_1"
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




