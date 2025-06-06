import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import numpy as np
import argparse
from tqdm.auto import tqdm
import sys

sys.path.append("/u1/n3jamali/3d-defense")


from models import *
from attacks.untargeted.drop import SaliencyDrop
from utils.utils import *
from dataset import ModelNet40Dataset, load_data


def attack():
    model.eval()
    all_adv_pc = []
    all_labels = []

    num = 0
    for pc, label in tqdm(test_loader):
        with torch.no_grad():
            pc, label = pc.float().cuda(), label.long().cuda()

        best_pc, success_num = attacker.attack(pc, label)

        num += success_num
        all_adv_pc += [best_pc]
        all_labels += [label.detach().cpu().numpy()]

    all_adv_pc = np.concatenate(all_adv_pc, axis=0)  
    all_labels = np.concatenate(all_labels, axis=0)

    return all_adv_pc, all_labels, num


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drop Attack')
    parser.add_argument('--data_root', type=str,
                        default='../data/ModelNet40')
    
    parser.add_argument('--model', type=str, default='pointnet',
                        choices=['pointnet', 'pointnet2', 'dgcnn', 'pct'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pct]')
    
    parser.add_argument('--feature_transform', type=str2bool, default=False,
                        help='whether to use STN on features in PointNet')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Size of batch)')
    
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')
    
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    
    parser.add_argument('--num_drop', type=int, default=200, metavar='N',
                        help='Number of dropping points')
    
    args = parser.parse_args()

    BEST_WEIGHTS = f'../checkpoints/{args.model.lower()}/{args.model.lower()}_best.pth'
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

    attacker = SaliencyDrop(model, num_drop=args.num_drop, alpha=1, k=5)

    test_transforms = data_transforms["test"][args.model.lower()]
    test_X, test_y = load_data(args.data_root, set="test", targeted=False)

    test_set = ModelNet40Dataset(test_X, test_y, transform=test_transforms)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    adv_pcs, gts, success = attack()

    data_num = len(test_set)
    acc = float(success) / float(data_num)

    print(acc)

    save_path = f"../data/adv_samples/drop{args.num_drop}/{args.model.lower()}/test"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.save(os.path.join(save_path, "pointclouds.npy"), adv_pcs)
    np.save(os.path.join(save_path, "labels.npy"), gts)

