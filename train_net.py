import os
import sys
import logging
import torch
import argparse
from defaults import get_cfg_defaults
from StyleGAN import train


def train_net(gpu_id, args):
    torch.cuda.set_device(0)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    if gpu_id == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        fh = logging.FileHandler(os.path.join(output_dir, 'log.txt'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.info(args)
    logger.info("Using {} GPUs".format(1))
    logger.info("Loaded configuration file {}".format(args.config_file))

    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train(cfg, logger)


if __name__ == '__main__':
    import os
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    parser = argparse.ArgumentParser(description="My Style")
    parser.add_argument(
        "--config-file",
        default="configs/experiment_celeba.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()
    train_net(gpu_id=0, args=args)