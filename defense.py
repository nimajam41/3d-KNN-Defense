# credits: https://github.com/Wuziyi616/IF-Defense/blob/main/baselines/defend_npz.py

import os
import tqdm
import argparse
import numpy as np
import torch

from defenses import SRSDefense, SORDefense, DUPNet

PU_NET_WEIGHT = 'defenses/DUP_Net/pu-in_1024-up_4.pth'


def defend(data_root, one_defense, targeted=True):
    test_path = os.path.join(data_root, "test")
    pc_path = os.path.join(test_path, "pointclouds.npy")
    label_path = os.path.join(test_path, "labels.npy")
    target_path = os.path.join(test_path, "targets.npy")

    save_path = os.path.join(data_root, one_defense, "test")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # data to defend
    batch_size = 128
    test_pc = np.load(pc_path)
    test_label = np.load(label_path)

    # defense module
    if one_defense.lower() == 'srs':
        defense_module = SRSDefense(drop_num=args.srs_drop_num)
        
    elif one_defense.lower() == 'sor':
        defense_module = SORDefense(k=args.sor_k, alpha=args.sor_alpha)

    elif one_defense.lower() == 'dup':
        up_ratio = 4
        defense_module = DUPNet(sor_k=args.sor_k,
                                sor_alpha=args.sor_alpha,
                                npoint=1024, up_ratio=up_ratio)
        
        defense_module.pu_net.load_state_dict(
            torch.load(PU_NET_WEIGHT))
        defense_module.pu_net = defense_module.pu_net.cuda()

    else:
        raise Exception("Undefined Defense!")

    # defend
    all_defend_pc = []

    for batch_idx in tqdm.trange(0, len(test_pc), batch_size):
        batch_pc = test_pc[batch_idx:batch_idx + batch_size]
        batch_pc = torch.from_numpy(batch_pc)

        if torch.cuda.is_available():
            batch_pc = batch_pc.float().cuda()

        defend_batch_pc = defense_module(batch_pc)

        # sor processed results have different number of points in each
        if isinstance(defend_batch_pc, list) or \
                isinstance(defend_batch_pc, tuple):
            defend_batch_pc = [
                pc.detach().cpu().numpy().astype(np.float32) for
                pc in defend_batch_pc
            ]

        else:
            defend_batch_pc = defend_batch_pc. \
                detach().cpu().numpy().astype(np.float32)
            defend_batch_pc = [pc for pc in defend_batch_pc]

        all_defend_pc += defend_batch_pc

    all_defend_pc = np.array(all_defend_pc, dtype="object")
    save_pc_path = os.path.join(save_path, "pointclouds.npy")
    save_labels_path = os.path.join(save_path, "labels.npy")
    save_target_path = os.path.join(save_path, "targets.npy")

    np.save(save_pc_path, all_defend_pc)
    np.save(save_labels_path, test_label)

    if targeted:
        test_target = np.load(target_path)
        np.save(save_target_path, test_target)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='3D Defense Methods')
    parser.add_argument('--data_root', type=str, default='data/ModelNet40',
                        help='Samples data root to defend')
    
    parser.add_argument('--defense', type=str, default='',
                        choices=['', 'srs', 'sor', 'dup'],
                        help='Defense method for input processing, '
                             'apply all if not specified')
    
    parser.add_argument('--srs_drop_num', type=int, default=500,
                        help='Number of point dropping in SRS')
    
    parser.add_argument('--sor_k', type=int, default=2,
                        help='KNN in SOR')
    
    parser.add_argument('--sor_alpha', type=float, default=1.1,
                        help='Threshold = mean + alpha * std')
    
    parser.add_argument('--targeted', type=str, default='true')
    args = parser.parse_args()

    # defense method
    if args.defense == '':
        all_defense = ['srs', 'sor', 'dup']
    else:
        all_defense = [args.defense]
        
    # targeted data
    if args.targeted.lower() == "true":
        targeted = True
    else:
        targeted = False

    # apply defense
    for one_defense in all_defense:
        print('{} defense'.format(one_defense))
        defend(args.data_root, one_defense=one_defense, targeted=targeted)
