"""
(Distributed) training script for scene segmentation
This file currently supports training and testing on S3DIS
If more than 1 GPU is provided, will launch multi processing distributed training by default
if you only wana using 1 GPU, set `CUDA_VISIBLE_DEVICES` accordingly
"""
import __init__
import argparse, yaml, os, logging, numpy as np
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist, multiprocessing as mp
from ammnet.utils import set_random_seed, save_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port
from ammnet.utils import AverageMeter, ConfusionMatrix, get_mious
from ammnet.dataset import build_dataloader_from_cfg
from ammnet.optim import build_optimizer_from_cfg
from ammnet.scheduler import build_scheduler_from_cfg
from ammnet.loss import build_criterion_from_cfg
from ammnet.models import build_model_from_cfg
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def main(gpu, cfg):
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank,
                                                    find_unused_parameters=True)
        logging.info('Using Distributed Data parallel ...')

    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    optimizer_disc = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)
    scheduler_disc = build_scheduler_from_cfg(cfg, optimizer)

    # build dataset
    val_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='val',
                                           distributed=cfg.distributed
                                           )
    logging.info(f"length of validation dataset: {len(val_loader.dataset)}")

    train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             )
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")

    # ===> start training
    val_miou, val_macc, val_oa, val_ious, val_accs = 0., 0., 0., [], []
    best_val, macc_when_best, oa_when_best, ious_when_best, best_epoch = 0., 0., 0., [], 0
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'epoch'):  # some dataset sets the dataset length as a fixed steps.
            train_loader.dataset.epoch = epoch - 1
        train2d_loss, train3d_loss, train_miou2d, train_miouSC, train_miouSSC, train_iousSSC = \
            train_one_epoch(model, train_loader, optimizer, optimizer_disc,
                            scheduler, scheduler_disc, epoch, cfg)

        is_best = False
        if epoch % cfg.val_freq == 0:
            val_miou2d, val_ious2d, val_miouSC, val_miouSSC, val_iousSSC, val_accsSSC = validate(model, val_loader, cfg)
            if val_miouSSC > best_val:
                is_best = True
                best_val = val_miouSSC
                macc_when_best = val_macc
                oa_when_best = val_oa
                ious_when_best = val_ious
                best_epoch = epoch
                with np.printoptions(precision=2, suppress=True):
                    logging.info(
                        f'Find a better ckpt @E{epoch}, val_miou2d {val_miou2d:.2f} \nval_ious2d: {val_ious2d}'
                        f'\nvalSC_miou {val_miouSC:.2f} valSSC_miou {val_miouSSC:.2f}'
                        f'\nval_iousSSC: {val_iousSSC}')

        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} LR {lr:.6f} '
                     f'train_miou {train_miou2d:.2f}, train_miouSC {train_miouSC:.2f}, '
                     f'train_miouSSC {train_miouSSC:.2f},'
                     f'val_miou {val_miou2d:.2f}, val_miouSC {val_miouSC:.2f}, '
                     f'val_miouSSC {val_miouSSC:.2f}, best val SSC miou {best_val:.2f}')

        if cfg.sched_on_epoch:
            scheduler.step(epoch)
        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val': best_val},
                            is_best=is_best)

    # validate
    with np.printoptions(precision=2, suppress=True):
        logging.info(
            f'Best ckpt @E{best_epoch},  val_oa {oa_when_best:.2f}, val_macc {macc_when_best:.2f}, val_miou {best_val:.2f}, '
            f'\niou per cls is: {ious_when_best}')
    return True


def train_disc_one_iter(model, data, optimizer_disc, scheduler_disc, num_iter, epoch, cfg):
    pred_3d, pred_2d, aug_info, loss_disc = model(**data, train_disc=True)
    loss = loss_disc * cfg.lossDisTrain
    loss.backward()
    if num_iter == cfg.step_per_update:
        if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip, norm_type=2)
        optimizer_disc.step()
        optimizer_disc.zero_grad()
        if not cfg.sched_on_epoch:
            scheduler_disc.step(epoch)


def train_one_epoch(model, train_loader, optimizer, optimizer_disc, scheduler, scheduler_disc, epoch, cfg):
    loss2d_meter, loss3d_meter, lossDis_meter = AverageMeter(), AverageMeter(), AverageMeter()
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    cmSC = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    cmSSC = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)

    model.train()  # set model to training mode

    pbar = tqdm(enumerate(train_loader), ncols=100, total=train_loader.__len__())
    num_iter = 0
    for idx, data in pbar:
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            if isinstance(data[key], str) or isinstance(data[key], list):
                continue
            data[key] = data[key].cuda(non_blocking=True)
        num_iter += 1

        train_disc_one_iter(model, data, optimizer_disc, scheduler_disc, num_iter, epoch, cfg)

        pred_3d, pred_2d, aug_info, loss_disc = model(**data)
        label_weight, label3d, mapping, mapping2d = data['label_weight'], data['label3d'], \
                                                    data['mapping'], data['mapping2d'].flatten(1)

        if aug_info['perm_xz']:
            pred_3d = pred_3d.permute(0, 1, 4, 3, 2)
        if aug_info['flip_z']:
            pred_3d = pred_3d.flip([4, ])
        if aug_info['flip_x']:
            pred_3d = pred_3d.flip([2, ])

        criterion = build_criterion_from_cfg(cfg.criterion).cuda()
        pred_2d = pred_2d.flatten(2).permute(0, 2, 1)

        pred_2d = torch.cat([pred_2d[i][mapping2d[i] != -1] for i in range(len(pred_2d))])
        label2d = torch.cat([label3d[i][mapping2d[i][mapping2d[i] != -1]] for i in range(len(label3d))])
        loss2d = criterion(pred_2d, label2d)

        '''
        loss 3d
        '''
        pred_3d = pred_3d.flatten(2).permute(0, 2, 1)

        weightSSC = label_weight & (label3d != cfg.ignore_index)
        pred_3dSSC, label3dSSC = pred_3d[weightSSC], label3d[weightSSC]
        loss3d = criterion(pred_3dSSC, label3dSSC)
        loss = loss2d * cfg.loss2d + loss3d + loss_disc * cfg.lossDis

        loss.backward()

        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0
            optimizer.step()
            optimizer.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        # update confusion matrix
        total_pixel = data['img'].shape[-2] * data['img'].shape[-1]
        weightSC = label_weight & (mapping == total_pixel) & (label3d != cfg.ignore_index)
        pred_3dSC, label3dSC = pred_3d[weightSC], label3d[weightSC]

        cm.update(pred_2d.argmax(dim=1), label2d)
        pred_3dSC, label3dSC = (pred_3dSC.argmax(dim=1) > 0).long(), (label3dSC > 0).long()
        cmSC.update(pred_3dSC, label3dSC)
        cmSSC.update(pred_3dSSC.argmax(dim=1), label3dSSC)
        loss2d_meter.update(loss2d.item())
        loss3d_meter.update(loss3d.item())
        lossDis_meter.update(loss_disc.item())

        if idx % cfg.print_freq:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss2d {loss2d_meter.val:.3f} Loss3d {loss3d_meter.val:.3f} "
                                 f"LossDis {lossDis_meter.val:.3f} Acc {cmSSC.overall_accuray:.2f}")
    miou, macc, oa, ious, accs = cm.all_metrics()
    miouSC, maccSC, oaSC, iousSC, accsSC = cmSC.all_metrics()
    miouSC = iousSC[1]
    miouSSC, maccSSC, oaSSC, iousSSC, accsSSC = cmSSC.all_metrics()
    miouSSC = np.mean(iousSSC[1:])
    return loss2d_meter.avg, loss3d_meter.avg, miou, miouSC, miouSSC, iousSSC


@torch.no_grad()
def validate(model, val_loader, cfg):
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    cmSC = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    cmSSC = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    for idx, data in pbar:
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            if isinstance(data[key], str) or isinstance(data[key], list):
                continue
            data[key] = data[key].cuda(non_blocking=True)

        pred_3d, pred_2d, _, _ = model(**data)

        label_weight, label3d, mapping = data['label_weight'], data['label3d'], data['mapping']
        pred_2d = pred_2d.flatten(2).permute(0, 2, 1)

        total_pixel = data['img'].shape[-2] * data['img'].shape[-1]
        pred_2d = torch.cat([pred_2d[i][mapping[i][mapping[i] != total_pixel]] for i in range(len(pred_2d))])
        label2d = torch.cat([label3d[i][mapping[i] != total_pixel] for i in range(len(label3d))])

        cm.update(pred_2d.argmax(dim=1), label2d)

        pred_3d = pred_3d.flatten(2).permute(0, 2, 1)
        weightSSC = label_weight & (label3d != cfg.ignore_index)
        pred_3dSSC, targetSSC = pred_3d[weightSSC], label3d[weightSSC]
        cmSSC.update(pred_3dSSC.argmax(dim=1), targetSSC)

        weightSC = label_weight & (data['mapping'] == total_pixel) & (label3d != cfg.ignore_index)
        pred_3dSC, targetSC = (pred_3d[weightSC].argmax(dim=1) > 0).long(), (label3d[weightSC] > 0).long()
        cmSC.update(pred_3dSC, targetSC)

    tp, union, count = cm.tp, cm.union, cm.count
    tpSC, unionSC, countSC = cmSC.tp, cmSC.union, cmSC.count
    tpSSC, unionSSC, countSSC = cmSSC.tp, cmSSC.union, cmSSC.count

    if cfg.distributed:
        dist.all_reduce(tp), dist.all_reduce(union), dist.all_reduce(count)
        dist.all_reduce(tpSC), dist.all_reduce(unionSC), dist.all_reduce(countSC)
        dist.all_reduce(tpSSC), dist.all_reduce(unionSSC), dist.all_reduce(countSSC)

    miou, macc, oa, ious, accs = get_mious(tp, union, count)
    miouSC, maccSC, oaSC, iousSC, accsSC = get_mious(tpSC, unionSC, countSC)
    miouSC = iousSC[1]
    miouSSC, maccSSC, oaSSC, iousSSC, accsSSC = get_mious(tpSSC, unionSSC, countSSC)
    miouSSC = np.mean(iousSSC[1:])
    return miou, ious, miouSC, miouSSC, iousSSC, accsSSC


if __name__ == "__main__":
    parser = argparse.ArgumentParser('semantic scene completion training/testing')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)    # overwrite the default arguments in yml 

    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)

    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]
    cfg.cfg_basename = args.cfg.split('.')[-2].split('/')[-1]
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.cfg_basename,  # cfg file name
        f'ngpus{cfg.world_size}',
        f'seed{cfg.seed}',
    ]
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            tags.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)

    cfg.is_training = cfg.mode in ['train', 'training', 'finetune', 'finetuning']
    if cfg.mode == 'train':
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
    else:  # resume from the existing ckpt and reuse the folder.
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path

    # multi processing.
    if cfg.mp:
        port = find_free_port()
        cfg.dist_url = f"tcp://localhost:{port}"
        print('using mp spawn for distributed training')
        mp.spawn(main, nprocs=cfg.world_size, args=(cfg, ))
    else:
        main(0, cfg)
