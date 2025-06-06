import os
import torch
import argparse
import sys
import time

sys.path.append("/u1/n3jamali/3d-defense")

from copy import deepcopy
from tqdm.auto import tqdm, trange
from models import *
from dataset import *
from utils import *

import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


def train(start_epoch):
    best_acc = 0
    best_epoch = 0
    best_weight = None

    for epoch in trange(start_epoch, args.epochs, desc='Epochs', leave=True):
        model.train()
        train_loss_save = 0
        acc_save = 0

        if args.model.lower() == 'pointnet':
            pointnet_loss_save = 0
            fea_loss_save = 0

        for batch_idx, data_pair in enumerate(tqdm(train_loader, desc='Batches', leave=False)):
            data, label = data_pair
            
            with torch.no_grad():
                data, label = data.float().cuda(), label.long().cuda()
                data = data.transpose(1, 2).contiguous()

            batch_size = data.size(0)
            optimizer.zero_grad()
            averages = {}

            if args.model.lower() == 'pointnet':
                logits, _, trans_feat = model(data)
                pointnet_loss = criterion(logits, label, False)

                if args.feature_transform:
                    fea_loss = feature_transform_reguliarzer(trans_feat) * 0.001
                else:
                    fea_loss = torch.tensor(0.).cuda()

                train_loss = pointnet_loss + fea_loss
                train_loss.backward()
                optimizer.step()

                train_loss_save += train_loss.item()
                pointnet_loss_save += pointnet_loss.item()
                fea_loss_save += fea_loss.item()

                acc = (torch.argmax(logits, dim=-1) ==label).sum().float() / float(batch_size)
                acc_save += acc.item()

                averages = {
                    "acc": acc_save / (batch_idx + 1),
                    "train": train_loss_save / (batch_idx + 1),
                    "pointnet": pointnet_loss_save / (batch_idx + 1),
                    "feat": fea_loss_save / (batch_idx + 1)
                }

                if (batch_idx + 1) % args.print_iter == 0:
                        print('Epoch {}, batch {}, lr: {:.6f}\n'
                        'Train loss: {:.4f}, PointNet loss: {:.4f}, Fea loss: {:.4f}\n'
                        'Train acc: {:.4f}'.
                        format(epoch, batch_idx + 1, get_lr(optimizer),
                                averages["train"], averages['pointnet'],
                                averages['feat'], averages['acc']))
                        
            
            else:
                logits = model(data)
                train_loss = criterion(logits, label, False)
                train_loss.backward()
                optimizer.step()

                train_loss_save += train_loss.item()
                acc = (torch.argmax(logits, dim=-1) ==label).sum().float() / float(batch_size)
                acc_save += acc.item()

                averages = {
                    "acc": acc_save / (batch_idx + 1),
                    "train": train_loss_save / (batch_idx + 1)
                }

                if (batch_idx + 1) % args.print_iter == 0:
                        print('Epoch {}, batch {}, lr: {:.6f}\n'
                        'Train loss: {:.4f}, Train acc: {:.4f}'.
                        format(epoch, batch_idx + 1, get_lr(optimizer),
                                averages["train"], averages['acc']))
                        
            
        if epoch % 10 == 0 or epoch > 180:
            acc = test()
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                best_weight = deepcopy(model.state_dict())
                
                torch.save(best_weight,
                           os.path.join(checkpoints_dir,
                                        '{}_best.pth'.
                                        format(args.model.lower())))

            print('Epoch {}, Test acc {:.4f}\nCurrent best acc {:.4f} at epoch {}'.
                format(epoch, acc, best_acc, best_epoch))
            
            torch.save(model.state_dict(),
                        os.path.join(
                        checkpoints_dir,
                        '{}_epoch{}_acc_{:.4f}_loss_{:.4f}_lr_{:.6f}.pth'.
                        format(args.model.lower(), epoch, acc, averages["train"], 
                                get_lr(optimizer))))
        
            torch.cuda.empty_cache()
            logger.add_scalar('test/acc', acc, epoch)

        
        logger.add_scalar('train/loss', averages["train"], epoch)
        logger.add_scalar('train/lr', get_lr(optimizer), epoch)

        scheduler.step(epoch)


def test():
    model.eval()
    acc_save = 0

    with torch.no_grad():
        for batch_idx, data_pair in enumerate(tqdm(test_loader, desc='Batches', leave=False)):
            data, label = data_pair
            data, label = data.float().cuda(), label.long().cuda()
            data = data.transpose(1, 2).contiguous()
            batch_size = data.size(0)

            if args.model.lower() == "pointnet":
                logits, _, _ = model(data)
            else:
                logits = model(data)

            preds = torch.argmax(logits, dim=-1)
            acc = (preds == label).sum().float() / float(batch_size)
            acc_save += acc.item()

            averages = {
                "acc": acc_save / (batch_idx + 1)
            }

        print('Test accuracy: {:.4f}'.format(averages["acc"]))
        return averages["acc"]


if __name__ == "__main__":
    print("Start!")
    parser = argparse.ArgumentParser(description='3D Point Clouds Classification Networks')
    parser.add_argument('--data_root', type=str,
                        default='data/ModelNet40')

    parser.add_argument('--model', type=str, default='pointnet',
                        choices=['pointnet', 'pointnet2', 'dgcnn', 'pct'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pct]')

    parser.add_argument('--feature_transform', type=str2bool, default=False,
                        help='whether to use STN on features in PointNet')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Size of batch)')

    parser.add_argument('--epochs', type=int, default=201, metavar='N',
                        help='Number of epochs to train ')

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='Learning rate for the optimizer')

    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')

    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')

    parser.add_argument('--print_iter', type=int, default=25,
                        help='Print interval')
                        
    args = parser.parse_args()
    set_seed(1)
    print(args)


    if args.model.lower() == 'dgcnn':
        model = DGCNN(args.emb_dims, args.k, output_channels=40)
    elif args.model.lower() == 'pointnet':
        model = PointNetCls(k=40, feature_transform=args.feature_transform)
    elif args.model.lower() == 'pointnet2':
        model = PointNet2ClsSsg(num_classes=40)
    elif args.model.lower() == "pct":
        model = PCT(output_channels=40)
    else:
        print('Model not recognized')
        exit(-1)

    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                     weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    train_transforms = data_transforms["train"][args.model.lower()]
    test_transforms = data_transforms["test"][args.model.lower()]

    train_X, train_y = load_data(args.data_root, set="train")
    test_X, test_y = load_data(args.data_root, set="test")


    train_set = ModelNet40Dataset(train_X, train_y, transform=train_transforms)
    test_set = ModelNet40Dataset(test_X, test_y, transform=test_transforms)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size * 2, shuffle=False)

    criterion = cal_loss

    start_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logs_dir = "logs/{}/{}".format(args.model, start_datetime)
    checkpoints_dir = "checkpoints/{}".format(args.model)

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    logger = SummaryWriter(os.path.join(logs_dir, 'logs'))
    train(1)