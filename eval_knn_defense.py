import sys

sys.path.append("/u1/n3jamali/3d-defense")

import torch
import numpy as np
import argparse
from models import *
from utils import *
from defenses.knn_defense import KNNDefense
from dataset import ModelNet40Dataset, load_data

from torchvision import transforms
from torch.utils.data import DataLoader


def extract_adv_features():
    model.eval()
    adv_features = []

    with torch.no_grad():
        for _, data_pair in enumerate(adv_loader):

            data, _ = data_pair
            data = data.float().cuda().transpose(1, 2).contiguous()

            _, features = model(data, return_global=True)
            adv_features += [feat.detach().cpu().numpy() for feat in features]

    
    return np.array(adv_features)


if __name__ == "__main__":
    print("Start!")
    parser = argparse.ArgumentParser(description='3D Point Clouds')
    parser.add_argument('--train_path', type=str,
                        default='data/ModelNet40')
    
    parser.add_argument('--adv_root', type=str,
                        default='data/adv_samples')
    
    parser.add_argument('--feature_root', type=str,
                        default='data/features')

    parser.add_argument('--model', type=str, default='pointnet',
                        choices=['pointnet', 'pointnet2', 'dgcnn', 'pct'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pct]')
    
    parser.add_argument('--attack', type=str, default='no_attack',
                        choices=['no_attack', 'shift', 'add_chamfer', 'add_hausdorff',
                                 'drop100', 'drop200', 'knn', 'uadvpc', 'tadvpc', 'uaof', 'taof'],
                        help='Attack name')
    
    parser.add_argument('--func', type=str, default='l2',
                        choices=['chamfer', 'hausdorff', 'l2'],
                        help='Distance function')

    parser.add_argument('--feature_transform', type=str2bool, default=False,
                        help='whether to use STN on features in PointNet')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='Size of batch)')
    
    parser.add_argument('--weight_func', type=str, default="UNIFORM",
                        choices=["UNIFORM", "CBWD", "CBWE"],
                        help='KNN-Defense weighting function')
    
    parser.add_argument('--num_nearest', type=int, default=5,
                        help="Num of nearest neighbors to use in KNN-Defense")

    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')

    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
                        
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

    test_transforms = data_transforms["test"][args.model.lower()]

    if args.func == "chamfer":
        nearest_func = fast_chamfer_distance

    elif args.func == "hausdorff":
        nearest_func = fast_hausdorff_distance
    
    elif args.func == "l2":
        nearest_func = euclidean_distance
    
    else:
        raise Exception("Undefined function!")
        
    
    adv_path = f"{args.adv_root}/{args.attack.lower()}/{args.model.lower()}"
    adv_samples, gt_labels = load_data(adv_path, set="test")
    adv_set = ModelNet40Dataset(adv_samples, gt_labels, transform=test_transforms)
    adv_loader = DataLoader(adv_set, batch_size=args.batch_size, shuffle=False)

    features_path = f"{args.feature_root}/{args.model.lower()}/features.npy"
    softmax_path = f"{args.feature_root}/{args.model.lower()}/softmax.npy"

    train_features = np.load(features_path)
    train_softmax = np.load(softmax_path)
    adv_features = extract_adv_features()

    dists = np.zeros((adv_features.shape[0], train_features.shape[0]))
    naive_dists = np.zeros((adv_features.shape[0], train_features.shape[0]))

    if args.func == "l2":
        dists = euclidean_distance(train_features, adv_features)

    else:
        # just for test, generally inefficient compared to L2 distance
        for i, adv_feat in enumerate(adv_features):
            for j, train_feat in enumerate(train_features):
                dists[i, j] = nearest_func(train_feat, adv_feat)

    train_labels = np.load(f"{args.train_path}/train/labels.npy")
    
    knn_defense = KNNDefense()
    preds, _ = knn_defense.defense(dists, train_softmax, k=args.num_nearest, weight=args.weight_func)

    acc = (preds == gt_labels).sum() / len(gt_labels)
    print("Accuaracy is: {:.4f}".format(100 * acc))