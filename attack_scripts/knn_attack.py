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
from attacks.targeted.knn import CWKNN
from attacks.utils import CrossEntropyAdvLoss, LogitsAdvLoss, ChamferkNNDist, ProjectInnerClipLinf
from utils.utils import *
from dataset import ModelNet40Dataset, load_data


def attack():
    model.eval()
    all_adv_pc = []
    all_labels = []
    all_targets = []

    num = 0
    for pc, label, target in tqdm(test_loader):
        with torch.no_grad():
            pc, label = pc.float().cuda(), label.long().cuda()
            target = target.long().cuda()

        best_pc, success_num = attacker.attack(pc, target)

        num += success_num
        all_adv_pc += [best_pc]
        all_labels += [label.detach().cpu().numpy()]
        all_targets += [target.detach().cpu().numpy()]

    all_adv_pc = np.concatenate(all_adv_pc, axis=0)  
    all_labels = np.concatenate(all_labels, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return all_adv_pc, all_labels, all_targets, num


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drop Attack')
    parser.add_argument('--data_root', type=str,
                        default='../data/ModelNet40')
    
    parser.add_argument('--model', type=str, default='pointnet',
                        choices=['pointnet', 'pointnet2', 'dgcnn', 'pct'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pct]')
    
    parser.add_argument('--feature_transform', type=str2bool, default=False,
                        help='whether to use STN on features in PointNet')

    parser.add_argument('--batch_size', type=int, default=-1,
                        help='Size of batch)')
    
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')
    
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    
    parser.add_argument('--adv_func', type=str, default='logits',
                        choices=['logits', 'cross_entropy'],
                        help='Adversarial loss function to use')
    
    parser.add_argument('--kappa', type=float, default=15.,
                        help='min margin in logits adv loss')
    
    parser.add_argument('--attack_lr', type=float, default=1e-3,
                        help='lr in CW optimization')
    
    parser.add_argument('--num_iter', type=int, default=2500, metavar='N',
                        help='Number of iterations in each search step')
    
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

    if args.adv_func == 'logits':
        adv_func = LogitsAdvLoss(kappa=args.kappa)
    else:
        adv_func = CrossEntropyAdvLoss()

    dist_func = ChamferkNNDist(chamfer_method='adv2ori',
                               knn_k=5, knn_alpha=1.05,
                               chamfer_weight=5., knn_weight=3.)
    clip_func = ProjectInnerClipLinf(budget=0.1)
    
    attacker = CWKNN(model, adv_func, dist_func, clip_func,
                     attack_lr=args.attack_lr,
                     num_iter=args.num_iter)

    test_X, test_y, test_target = load_data(args.data_root, set="test", targeted=True)

    test_set = ModelNet40Dataset(test_X, test_y, transform=to_tensor_transform, t=test_target)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    adv_pcs, gts, targets, success = attack()

    data_num = len(test_set)
    acc = float(success) / float(data_num)

    print(acc)

    save_path = f"../data/adv_samples/knn/{args.model.lower()}/test"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.save(os.path.join(save_path, "pointclouds.npy"), adv_pcs)
    np.save(os.path.join(save_path, "labels.npy"), gts)
    np.save(os.path.join(save_path, "targets.npy"), targets)

