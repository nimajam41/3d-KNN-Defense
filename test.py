import os
import torch
import argparse
import sys
import time

sys.path.append("/u1/n3jamali/3d-defense")

from models import *
from dataset import *
from utils import *
from torch.utils.data import DataLoader


def test():
    model.eval()

    total = 0.
    correct = 0.

    with torch.no_grad():
        

        for batch_idx, data_pair in enumerate(test_loader):

            data, label = data_pair
            data, label = data.float().cuda(), label.long().cuda()

            data = data.transpose(1, 2).contiguous()
            batch_size = data.size(0)
            total += batch_size

            if args.model.lower() == "pointnet":
                logits, _, _ = model(data)
            else:
                logits = model(data)

            preds = torch.argmax(logits, dim=-1)
            correct += (preds == label).sum().float()


        return correct / total


if __name__ == "__main__":
    print("Start!")
    parser = argparse.ArgumentParser(description='3D Point Clouds')
    parser.add_argument('--data_root', type=str,
                        default='data/ModelNet40')

    parser.add_argument('--model', type=str, default='pointnet',
                        choices=['pointnet', 'pointnet2', 'dgcnn', "pct"],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pct]')

    parser.add_argument('--feature_transform', type=str2bool, default=False,
                        help='whether to use STN on features in PointNet')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='Size of batch)')

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='Learning rate for the optimizer')

    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')

    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    
    parser.add_argument('--apply_transform', type=str2bool, default=False,
                        help="whether to apply normalization and other test transforms")

                        
    args = parser.parse_args()
    BEST_WEIGHTS = f'checkpoints/{args.model.lower()}/{args.model.lower()}_best.pth'

    set_seed(1)
    print(args)


    if args.model.lower() == 'dgcnn':
        model = DGCNN(args.emb_dims, args.k, output_channels=40)

    elif args.model.lower() == 'pointnet':
        model = PointNetCls(k=40, feature_transform=args.feature_transform)

    elif args.model.lower() == 'pointnet2':
        model = PointNet2ClsSsg(num_classes=40)

    elif args.model.lower() == 'pct':
        model = PCT(output_channels=40)

    else:
        print('Model not recognized')
        exit(-1)

    model = model.cuda()
    print('Loading weight {}'.format(BEST_WEIGHTS))
    state_dict = torch.load(BEST_WEIGHTS)

    try:
        model.load_state_dict(state_dict)

    except RuntimeError:
        model.module.load_state_dict(state_dict)

    test_transforms = to_tensor_transform
    if args.apply_transform:
        test_transforms = data_transforms["test"][args.model.lower()]

    test_X, test_y = load_data(args.data_root, set="test")
    test_set = ModelNet40Dataset(test_X, test_y, transform=test_transforms)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    acc = test()
    print("Accuaracy is: {:.4f}".format(100 * acc))