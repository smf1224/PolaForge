import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import argparse
import datetime
import random
import time
from pathlib import Path
import numpy as np
import torch
import util.misc as utils
from engine import train_one_epoch
from models import build_model
from datasets import create_dataset
import cv2
from eval.evaluate import eval
from util.logger import get_logger
from tqdm import tqdm
from util.PolyLR import LinoPolyScheduler
from mmengine.optim.scheduler.lr_scheduler import PolyLR
from util.misc import (NestedTensor, nested_tensor_from_tensor_list)


def get_args_parser():

    parser = argparse.ArgumentParser('PolaForge', add_help=False)
    parser.add_argument('--BCELoss_ratio', default=0.45, type=float)
    parser.add_argument('--DiceLoss_ratio', default=0.55, type=float)
    parser.add_argument('--dataset_path', default="/home/b311/data1/25-lijiayu/temp/tii/data/NEU", help='path to images')
    # parser.add_argument('--dataset_path', default="../data/poldata", help='path to images')
    parser.add_argument('--batch_size_train', type=int, default=1, help='train input batch size')
    parser.add_argument('--batch_size_test', type=int, default=1, help='test input batch size')
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
    parser.add_argument('--Norm_Type', type=str, default='GN', help='Norm Type')
    parser.add_argument('--modals', nargs='+', default=['RGB', 'dep'], help='modal name for process')
    parser.add_argument('--lr_scheduler', type=str, default='PolyLR')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop', default=30, type=int)
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--output_dir', default='./checkpoints/weights', help='save path for weights')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--dataset_mode', type=str, default='crack')
    parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches,  takes them randomly')
    parser.add_argument('--num_threads', default=1, type=int, help='num_workers')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    parser.add_argument('--load_width', type=int, default=512, help='load image width')
    parser.add_argument('--load_height', type=int, default=512, help='load image height')
    parser.add_argument('--num_queries', default=1024, type=int, help="Number of query slots")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    parser.add_argument('--hidden_dim', default=256, type=int, help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--d_state', default=16, type=int)
    parser.add_argument('--d_conv', default=4, type=int)
    parser.add_argument('--expand', default=2, type=int)
    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    parser.add_argument('--patch_size', type=int, default=8)

    return parser


def main(args):
    checkpoints_path = "./checkpoints"
    curTime = time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime(time.time()))
    dataset_name = (args.dataset_path).split('/')[-1]
    modal_num = len(args.modals)

    modals_name = ''
    for i in range(len(args.modals)):
        modals_name = modals_name + '_' + (args.modals)[i]

    process_floder_path = os.path.join(checkpoints_path, curTime + '_Dataset->' + dataset_name + '_modals->' + modals_name)
    if not os.path.exists(process_floder_path):
        os.makedirs(process_floder_path)
    else:
        print("create process floder error!")

    log_train = get_logger(process_floder_path, 'train')
    log_test = get_logger(process_floder_path, 'test')
    log_eval = get_logger(process_floder_path, 'eval')


    # print(args)
    log_train.info("args -> " + str(args))
    log_train.info("args: dataset -> " + str(args.dataset_path))
    log_train.info("processing modal -> " + str(args.modals))
    log_train.info("number of modal -> " + str(modal_num))
    print('processing modal -> ', args.modals)
    print('number of modal -> ', modal_num)


    device = torch.device(args.device)

    # 固定随机数
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion = build_model(args)
    model.to(device)

    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('number of params:', n_parameters)

    args.batch_size = args.batch_size_train
    train_dataLoader = create_dataset(args)
    dataset_size = len(train_dataLoader)
    print('The number of training images = %d' % dataset_size)
    log_train.info('The number of training images = %d' % dataset_size)


    param_dicts = [
        {
            "params":
                [p for n, p in model.named_parameters()],
            "lr": args.lr,
        },
    ]
    if args.sgd:
        print('use SGD!')
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        print('use AdamW!')
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)

    lr_scheduler = None
    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    elif args.lr_scheduler == 'CosLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-5)
    elif args.lr_scheduler == 'PolyLR':
        lr_scheduler = PolyLR(optimizer, eta_min=args.min_lr, begin=args.start_epoch, end=args.epochs)
        # lr_scheduler = LinoPolyScheduler(optimizer, min_lr=args.min_lr, epochs=args.epochs)

    output_dir = Path(args.output_dir)

    print("Start processing! ")
    log_train.info("Start processing! ")
    start_time = time.time()
    max_F1 = 0
    max_mIoU = 0
    max_Metrics = {'epoch': 0, 'mIoU': 0, 'ODS': 0, 'OIS': 0, 'F1': 0, 'Precision': 0, 'Recall': 0}
    max_mIoU_Metrics = {'epoch': 0, 'mIoU': 0, 'ODS': 0, 'OIS': 0, 'F1': 0, 'Precision': 0, 'Recall': 0}

    for epoch in range(args.start_epoch, args.epochs):
        args.phase = 'train'
        print("---------------------------------------------------------------------------------------")
        print("training epoch start -> ", epoch)

        train_one_epoch(
            model, criterion, train_dataLoader, optimizer, epoch, args, log_train)

        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        print("training epoch finish -> ", epoch)
        print("---------------------------------------------------------------------------------------")


        # 测试
        print("testing epoch start -> ", epoch)


        results_path = curTime + '_Dataset->' + dataset_name + '_modal->' + modals_name
        save_root = f'./results/{results_path}/results_' + str(epoch)
        args.phase = 'test'
        args.batch_size = args.batch_size_test
        test_dl = create_dataset(args)  # create a dataset given opt.dataset_mode and other options
        pbar = tqdm(total=len(test_dl), desc=f"Initial Loss: Pending")

        if not os.path.isdir(save_root):
            os.makedirs(save_root)
        with torch.no_grad():
            for batch_idx, (data) in enumerate(test_dl):

                mask = data['label'].squeeze(1).bool()
                modal_imgs = []
                for i in range(modal_num):
                    items = list(data.items())
                    key, value = items[i]
                    value = NestedTensor(value, mask).to(torch.device(args.device))
                    modal_imgs.append(value.to(torch.device(args.device)))

                target = data["label"]
                target = target.to(dtype=torch.int64).cuda()
                # _, _, out = model(x)
                out = model(modal_imgs)

                loss = criterion(out, target.float())

                target = target[0, 0, ...].cpu().numpy()
                out = out[0, 0, ...].cpu().numpy()
                root_name = data["image_path"][0].split("/")[-1][0:-4]

                target = 255 * (target / np.max(target))
                out = 255 * (out / np.max(out))

                # out[out >= 0.3] = 255
                # out[out < 0.3] = 0

                log_test.info('----------------------------------------------------------------------------------------------')
                log_test.info("loss -> " + str(loss))
                log_test.info(str(os.path.join(save_root, "{}_lab.png".format(root_name))))
                log_test.info(str(os.path.join(save_root, "{}_pre.png".format(root_name))))
                log_test.info('----------------------------------------------------------------------------------------------')
                cv2.imwrite(os.path.join(save_root, "{}_lab.png".format(root_name)), target)
                cv2.imwrite(os.path.join(save_root, "{}_pre.png".format(root_name)), out)
                pbar.set_description(f"Loss: {loss.item():.4f}")
                pbar.update(1)  # Manually increment the progress bar by one

        pbar.close()
        log_test.info("model -> " + str(epoch) + " test finish!")
        log_test.info('----------------------------------------------------------------------------------------------')
        print("testing epoch finish -> ", epoch)

        print("---------------------------------------------------------------------------------------")

        print("evalauting epoch start -> ", epoch)
        metrics = eval(log_eval, save_root, epoch)
        for key, value in metrics.items():
            print(str(key) + ' -> ' + str(value))
        if(max_F1 < metrics['F1']):
            max_Metrics = metrics
            max_F1 = metrics['F1']
            checkpoint_paths = [output_dir / f'checkpoint_best.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
            log_train.info("\nupdate and save best model -> " + str(epoch))
            print("\nupdate and save best model -> ", epoch)

        # 如果当前 mIoU 更大，更新记录
        if metrics['mIoU'] > max_mIoU:
            max_mIoU = metrics['mIoU']
            max_mIoU_Metrics = metrics.copy()

            # 可选：保存最大 mIoU 对应模型
            checkpoint_paths = [output_dir / f'checkpoint_best_mIoU.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
            log_train.info("\nupdate and save best mIoU model -> " + str(epoch))
            print("\nupdate and save best mIoU model -> ", epoch)

        print("evalauting epoch finish -> ", epoch)
        print('\nmax_F1 -> ' + str(max_Metrics['F1']) + '\nmax Epoch -> ' + str(max_Metrics['epoch']))
        print('\nmax_mIoU -> ' + str(max_Metrics['mIoU']) + '\nmax Epoch -> ' + str(max_Metrics['epoch']))
        print("---------------------------------------------------------------------------------------")

        log_eval.info("evalauting epoch finish -> " + str(epoch))
        log_eval.info('\nmax_F1 -> ' + str(max_Metrics['F1']) + '\nmax Epoch -> ' + str(max_Metrics['epoch']))
        log_eval.info('\nmax_mIoU -> ' + str(max_Metrics['mIoU']) + '\nmax Epoch -> ' + str(max_Metrics['epoch']))
        log_eval.info("---------------------------------------------------------------------------------------")


    for key, value in max_Metrics.items():
        log_eval.info(str(key) + ' -> ' + str(value))
    log_eval.info('\nmax_F1 -> ' + str(max_Metrics['F1']) + '\nmax Epoch -> ' + str(max_Metrics['epoch']))
    log_eval.info('\nmax_mIoU -> ' + str(max_mIoU_Metrics['mIoU']) + '\nmax Epoch -> ' + str(max_mIoU_Metrics['epoch']))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Process time {}'.format(total_time_str))
    log_train.info('Process time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PolaForge', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

