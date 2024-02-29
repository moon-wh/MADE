import logging
import os
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval,R1_mAP_eval_LTCC
from torch.cuda import amp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             local_rank,
             dataset,
             val_loader = None,
             val_loader_same = None):
    log_period = cfg.SOLVER.LOG_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("EVA-attribure.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],find_unused_parameters=True)

    train_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'train'))
    rank_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'rank'))
    mAP_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'mAP'))
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    if cfg.DATA.DATASET == 'ltcc':
        evaluator_diff = R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)  # ltcc
        evaluator_general = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    elif cfg.DATA.DATASET == 'prcc':
        evaluator_diff = R1_mAP_eval(dataset.num_query_imgs_diff, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)  # prcc
        evaluator_same = R1_mAP_eval(dataset.num_query_imgs_same, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)


    scaler = amp.GradScaler()
    best_rank1 = -np.inf
    best_epoch = 0
    start_train_time = time.time()
    # train
    for epoch in range(cfg.TRAIN.START_EPOCH, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()

        if cfg.DATA.DATASET == 'ltcc':
            evaluator_diff.reset()
            evaluator_general.reset()

        elif cfg.DATA.DATASET == 'prcc':
            evaluator_diff.reset()
            evaluator_same.reset()
        else:
            evaluator.reset()

        scheduler.step(epoch)
        model.train()
        for idx, data in enumerate(train_loader):
            if cfg.MODEL.ADD_META:
                print("--------- here")
                samples, targets, camids, _,clothes, meta = data
                print("--------- meta",meta.shape)
                meta = [m.float() for m in meta]
                meta = torch.stack(meta, dim=0)
                meta = meta.cuda(non_blocking=True)
                if cfg.MODEL.MASK_META:
                    meta[:, 5:21] = 0
                    meta[:, 35:57] = 0
                    meta[:, 84] = 0
                    meta[:, 90] = 0
                    meta[:, 92:96] = 0
                    meta[:, 97] = 0
                    meta[:, 100] = 0
                    meta[:, 102:105] = 0
            else:
                samples, targets, camids,_, clothes,meta,text = data
                meta = None

            samples = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            clothes = clothes.cuda(non_blocking=True)
            print("--------- samples",samples.shape)
            print("--------- clothes",clothes.shape)
            print("--------- targets",targets.shape)
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            with amp.autocast(enabled=True):
                if cfg.MODEL.ADD_META:
                    if cfg.MODEL.CLOTH_ONLY:
                        score, feat = model(samples, clothes)
                    else:
                        score, feat = model(samples, clothes, meta)

                else:
                    score, feat = model(samples)
            print("--------- score",score.shape)
            print("--------- feat",feat.shape)
            print("--------- camids",camids.shape)
            loss = loss_fn(score, feat, targets, camids)
            print("--------- loss",loss.item())
            print("#############################\n")
            train_writer.add_scalar('loss', loss.item(), epoch)
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == targets).float().mean()
            else:
                acc = (score.max(1)[1] == targets).float().mean()

            loss_meter.update(loss.item(), samples.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (idx + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (idx + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (idx + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % eval_period == 0:
            model.eval()
            if cfg.DATA.DATASET == 'prcc':
                logger.info("Clothes changing setting")
                rank1= test(cfg, model, evaluator_diff, val_loader, logger, device,epoch, rank_writer, mAP_writer)
                logger.info("Standard setting")
                test(cfg, model, evaluator_same, val_loader_same, logger, device, epoch,  rank_writer, mAP_writer,test=True)
            elif cfg.DATA.DATASET == 'ltcc':
                logger.info("Clothes changing setting")
                rank1 = test(cfg, model, evaluator_diff, val_loader, logger, device,epoch,rank_writer, mAP_writer, cc=True)
                logger.info("Standard setting")
                test(cfg, model, evaluator_general, val_loader, logger, device, epoch, rank_writer, mAP_writer,test=True)
            else:
                rank1= test(cfg, model, evaluator, val_loader, logger, device,epoch,rank_writer, mAP_writer)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch
                logger.info("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))
                if cfg.MODEL.DIST_TRAIN:
                    if dist.get_rank() == 0:
                        torch.save(model.state_dict(),
                                   os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))

    if cfg.MODEL.DIST_TRAIN:
        if dist.get_rank() == 0:
            logger.info("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    total_time = time.time() - start_train_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def do_inference(cfg,
                 model,
                 dataset,
                 val_loader = None,
                 val_loader_same=None):
    logger = logging.getLogger("EVA-attribure.test")
    logger.info("Enter inferencing")

    logger.info("transreid inferencing")
    device = "cuda"
    if cfg.DATA.DATASET == 'ltcc':
        evaluator_diff = R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)  # ltcc
        evaluator_general = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    elif cfg.DATA.DATASET == 'prcc':
        evaluator_diff = R1_mAP_eval(dataset.num_query_imgs_diff, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)  # prcc
        evaluator_same = R1_mAP_eval(dataset.num_query_imgs_same, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    if cfg.DATA.DATASET == 'ltcc':
        evaluator_diff.reset()
        evaluator_general.reset()

    elif cfg.DATA.DATASET == 'prcc':
        evaluator_diff.reset()
        evaluator_same.reset()
    else:
        evaluator.reset()
    model.to(device)
    model.eval()
    if cfg.DATA.DATASET == 'prcc':
        logger.info("Clothes changing setting")
        test(cfg, model, evaluator_diff, val_loader, logger, device, test=True)
        logger.info("Standard setting")
        test(cfg, model, evaluator_same, val_loader_same, logger, device, test=True)
    elif cfg.DATA.DATASET == 'ltcc':
        logger.info("Clothes changing setting")
        test(cfg, model, evaluator_diff, val_loader, logger, device, test=True,cc=True)
        logger.info("Standard setting")
        test(cfg, model, evaluator, val_loader, logger, device, test=True)
    else:
        test(cfg, model, evaluator, val_loader, logger, device, test=True)

def test(cfg, model, evaluator, val_loader, logger, device, epoch=None, rank_writer=None, mAP_writer=None,test=False,cc=False):
    for n_iter, (imgs, pids, camids, clothes_id, clothes_ids, meta) in enumerate(val_loader):
        with torch.no_grad():
            imgs = imgs.to(device)
            meta = meta.to(device)
            clothes_ids = clothes_ids.to(device)
            meta = meta.to(torch.float32)
            if cfg.MODEL.CLOTH_ONLY:
                feat = model(imgs, clothes_ids)
            else:
                if cfg.MODEL.MASK_META:
                    meta[:, 5:21] = 0
                    meta[:, 35:57] = 0
                    meta[:, 84] = 0
                    meta[:, 90] = 0
                    meta[:, 92:96] = 0
                    meta[:, 97] = 0
                    meta[:, 100] = 0
                    meta[:, 102:105] = 0
                if cfg.TEST.TYPE == 'image_only':
                    meta = torch.zeros_like(meta)
                feat = model(imgs, clothes_ids, meta)
            if cc:
                evaluator.update((feat, pids, camids, clothes_id))
            else:
                evaluator.update((feat, pids, camids))
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    if test :
        torch.cuda.empty_cache()
        return
    logger.info("Validation Results - Epoch: {}".format(epoch))
    rank1 = cmc[0]
    rank_writer.add_scalar('rank1', rank1, epoch)
    mAP_writer.add_scalar('mAP', mAP, epoch)
    torch.cuda.empty_cache()
    return rank1




