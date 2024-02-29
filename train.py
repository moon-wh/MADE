from utils.logger import setup_logger
from data import build_dataloader
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
from model import build_model
# from utils import auto_resume_helper,load_checkpoint

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="CC_ReID  Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger("EVA-attribure", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    if cfg.DATA.DATASET == 'prcc':
        trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler,val_loader,val_loader_same= build_dataloader(
            cfg)  # prcc_test
    else:
        trainloader, queryloader, galleryloader, dataset, train_sampler,val_loader = build_dataloader(cfg)

    model = build_model(cfg,dataset.num_train_pids,dataset.num_train_clothes)

    loss_func, center_criterion = make_loss(cfg, num_classes=dataset.num_train_pids)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)

    if cfg.DATA.DATASET == 'prcc':
        do_train(
            cfg,
            model,
            center_criterion,
            trainloader,
            optimizer,
            optimizer_center,
            scheduler,
            loss_func,
            args.local_rank,
            dataset,
            val_loader=val_loader,
            val_loader_same=val_loader_same
        )
    else:
        do_train(
            cfg,
            model,
            center_criterion,
            trainloader,
            optimizer,
            optimizer_center,
            scheduler,
            loss_func,
            args.local_rank,
            dataset,
            val_loader=val_loader
        )
