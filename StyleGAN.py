import torch.utils.data
from torch import optim
from torchvision.utils import save_image
import torch.cuda.comm
import torch.cuda.nccl
import dlutils.pytorch.count_parameters as count_param_override
import module.lod_driver as lod_driver
from dataUtils.dataloader import * #TFRecordsDataset,make_dataloader
from module.model import Model
from module.net import *
import nativeUtils.utils as utils
from nativeUtils.tracker import LossTracker
from nativeUtils.checkpointer import Checkpointer
from nativeUtils.scheduler import ComboMultiStepLR
from module.custom_adam import LREQAdam
from tqdm import tqdm
from defaults import get_cfg_defaults

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
#训练函数 和 保存图像函数

def save_sample(lod2batch, tracker, sample, x, logger, model, cfg, discriminator_optimizer, generator_optimizer):
    os.makedirs('results', exist_ok=True)

    logger.info('\n[%d/%d] - ptime: %.2f, %s, blend: %.3f lr: %.12f,  %.12f, max mem: %f",' % (
        lod2batch.current_epoch, cfg.TRAIN.TRAIN_EPOCHS, lod2batch.per_epoch_ptime, str(tracker),
        lod2batch.get_blend_factor(),
        discriminator_optimizer.param_groups[0]['lr'],
        generator_optimizer.param_groups[0]['lr'],
        torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))

    with torch.no_grad():
        model.eval()
        x_rec = model.generate(lod2batch.lod, lod2batch.get_blend_factor(), z=sample)

        @utils.async_func
        def save_pic(x, x_rec):
            tracker.register_means(lod2batch.current_epoch + lod2batch.iteration * 1.0 / lod2batch.get_dataset_size())
            tracker.plot()

            #x_rec = F.interpolate(x_rec, 128)
            result_sample = x_rec * 0.5 + 0.5
            result_sample = result_sample.cpu()
            save_image(result_sample, os.path.join(cfg.OUTPUT_DIR,
                                                   'sample_%d_%d.jpg' % (
                                                       lod2batch.current_epoch + 1,
                                                       lod2batch.iteration // 1000)
                                                   ), nrow=16)

        save_pic(x, x_rec)


def train(cfg, logger, gpu_id=0):
    torch.cuda.set_device(gpu_id)
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        dlatent_avg_beta=cfg.MODEL.DLATENT_AVG_BETA,
        style_mixing_prob=cfg.MODEL.STYLE_MIXING_PROB,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=3)
    model.cuda(gpu_id)
    model.train()

    if gpu_id == 0:
        model_s = Model(
            startf=cfg.MODEL.START_CHANNEL_COUNT,
            layer_count=cfg.MODEL.LAYER_COUNT,
            maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
            latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
            truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
            truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
            mapping_layers=cfg.MODEL.MAPPING_LAYERS,
            channels=3)
        del model_s.discriminator
        model_s.cuda(gpu_id)
        model_s.eval()
        model_s.requires_grad_(False)

    generator = model.generator
    discriminator = model.discriminator
    mapping = model.mapping
    dlatent_avg = model.dlatent_avg

    count_param_override.print = lambda a: logger.info(a)

    logger.info("Trainable parameters generator:")
    count_parameters(generator)

    logger.info("Trainable parameters discriminator:")
    count_parameters(discriminator)

    generator_optimizer = LREQAdam([
        {'params': generator.parameters()},
        {'params': mapping.parameters()}], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)

    discriminator_optimizer = LREQAdam([
        {'params': discriminator.parameters()},
    ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)

    scheduler = ComboMultiStepLR(optimizers={'generator': generator_optimizer,'discriminator': discriminator_optimizer},
                                 milestones=cfg.TRAIN.LEARNING_DECAY_STEPS, # []
                                 gamma=cfg.TRAIN.LEARNING_DECAY_RATE, # 0.1
                                 reference_batch_size=32, base_lr=cfg.TRAIN.LEARNING_RATES) # 0.002

    model_dict = {
        'discriminator': discriminator,
        'generator': generator,
        'mapping': mapping,
        'dlatent_avg': dlatent_avg
    }

    if gpu_id == 0:
        model_dict['generator_s'] = model_s.generator
        model_dict['mapping_s'] = model_s.mapping

    tracker = LossTracker(cfg.OUTPUT_DIR)

    checkpointer = Checkpointer(cfg,
                                model_dict,
                                {
                                    'generator_optimizer': generator_optimizer,
                                    'discriminator_optimizer': discriminator_optimizer,
                                    'scheduler': scheduler,
                                    'tracker': tracker
                                },
                                logger=logger,
                                save=gpu_id == 0)

    checkpointer.load()
    logger.info("Starting from epoch: %d" % (scheduler.start_epoch()))

    layer_to_resolution = generator.layer_to_resolution #[4, 8, 16, 32, 64, 128]

    dataset = TFRecordsDataset(cfg, logger, buffer_size_mb=1024)

    rnd = np.random.RandomState(3456)
    latents = rnd.randn(32, cfg.MODEL.LATENT_SPACE_SIZE)
    sample = torch.tensor(latents).float().cuda()

    lod2batch = lod_driver.LODDriver(cfg, logger, gpu_num=1, dataset_size=len(dataset)) #一个可以返回各类训练参数(param)的对象

    for epoch in range(scheduler.start_epoch(), cfg.TRAIN.TRAIN_EPOCHS):
        model.train()
        lod2batch.set_epoch(epoch, [generator_optimizer, discriminator_optimizer])

        logger.info("Batch size: %d, Batch size per GPU: %d, LOD: %d - %dx%d, blend: %.3f, dataset size: %d" % (
                                                                lod2batch.get_batch_size(),
                                                                lod2batch.get_per_GPU_batch_size(),
                                                                lod2batch.lod,
                                                                2 ** lod2batch.get_lod_power2(),
                                                                2 ** lod2batch.get_lod_power2(),
                                                                lod2batch.get_blend_factor(),
                                                                len(dataset)))

        dataset.reset(lod2batch.get_lod_power2(), lod2batch.get_per_GPU_batch_size())
        batches = make_dataloader(cfg, logger, dataset, lod2batch.get_per_GPU_batch_size(), gpu_id) # 一个数据集分为多个batch,一个batch有n长图片

        scheduler.set_batch_size(lod2batch.get_batch_size(), lod2batch.lod) #报错！

        need_permute = False

        with torch.autograd.profiler.profile(use_cuda=True, enabled=False) as prof:
            for x_orig in tqdm(batches): # x_orig:[-1,c,w,h]

                with torch.no_grad():
                    if x_orig.shape[0] != lod2batch.get_per_GPU_batch_size():
                        continue
                    if need_permute:
                        x_orig = x_orig.permute(0, 3, 1, 2)
                    x_orig = (x_orig / 127.5 - 1.)

                    blend_factor = lod2batch.get_blend_factor()

                    needed_resolution = layer_to_resolution[lod2batch.lod]
                    x = x_orig

                    if lod2batch.in_transition:
                        needed_resolution_prev = layer_to_resolution[lod2batch.lod - 1]
                        x_prev = F.avg_pool2d(x_orig, 2, 2)
                        x_prev_2x = F.interpolate(x_prev, needed_resolution)
                        x = x * blend_factor + x_prev_2x * (1.0 - blend_factor)
                x.requires_grad = True

                discriminator_optimizer.zero_grad()
                loss_d = model(x, lod2batch.lod, blend_factor, d_train=True)
                tracker.update(dict(loss_d=loss_d))
                loss_d.backward()
                discriminator_optimizer.step()

                if gpu_id == 0:
                    betta = 0.5 ** (lod2batch.get_batch_size() / (10 * 1000.0))
                    model_s.lerp(model, betta)

                generator_optimizer.zero_grad()
                loss_g = model(x, lod2batch.lod, blend_factor, d_train=False)
                tracker.update(dict(loss_g=loss_g))
                loss_g.backward()
                generator_optimizer.step()

                lod2batch.step()
                if gpu_id == 0:
                    if lod2batch.is_time_to_save():
                        checkpointer.save("model_tmp_intermediate")
                    if lod2batch.is_time_to_report():
                        save_sample(lod2batch, tracker, sample, x, logger, model_s, cfg, discriminator_optimizer, generator_optimizer)
        scheduler.step()

        if gpu_id == 0:
            checkpointer.save("model_tmp")
            save_sample(lod2batch, tracker, sample, x, logger, model_s, cfg, discriminator_optimizer, generator_optimizer)

    logger.info("Training finish!... save training results")
    if gpu_id == 0:
        checkpointer.save("model_final").wait()


if __name__ == "__main__":
    gpu_count = torch.cuda.device_count()
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
