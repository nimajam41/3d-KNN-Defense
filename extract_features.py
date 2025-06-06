import os
import torch
import argparse
import sys
from torch.utils.data import DataLoader
from torch.nn.functional import softmax

sys.path.append("/u1/n3jamali/3d-defense")

from models import *
from dataset import *
from utils import *


def extract():
    model.eval()

    all_softmax = []
    train_features = []

    softmax_path = os.path.join(save_path, "softmax.npy")
    features_path = os.path.join(save_path, "features.npy")

    with torch.no_grad():

        for _, data_pair in enumerate(test_loader):

            data, _ = data_pair
            data = data.float().cuda().transpose(1, 2).contiguous()

            logits, features = model(data, return_global=True)
            s = softmax(logits, dim=1)

            all_softmax += [soft.detach().cpu().numpy() for soft in s]
            train_features += [feat.detach().cpu().numpy() for feat in features]

    
    all_softmax = np.array(all_softmax)
    train_features = np.array(train_features)

    np.save(softmax_path, all_softmax)
    np.save(features_path, train_features)


if __name__ == "__main__":
    print("Start!")
    parser = argparse.ArgumentParser(description='3D Point Clouds')
    parser.add_argument('--data_root', type=str,
                        default='data/ModelNet40')

    parser.add_argument('--model', type=str, default='pointnet',
                        choices=['pointnet', 'pointnet2', 'dgcnn', 'curvenet', 'pct'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, curvenet, pct]')

    parser.add_argument('--feature_transform', type=str2bool, default=False,
                        help='whether to use STN on features in PointNet')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='Size of batch)')

    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')

    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    
    parser.add_argument('--apply_transform', type=str2bool, default=True,
                        help="whether to apply normalization and other test transforms")

                        
    args = parser.parse_args()
    BEST_WEIGHTS = f'checkpoints/{args.model.lower()}/{args.model.lower()}_best.pth'

    set_seed(1)
    print(args)

    save_path = os.path.join("data/features", args.model.lower())
    if not os.path.exists(save_path):
        os.makedirs(save_path)

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

    train_X, train_y = load_data(args.data_root, set="train")
    train_set = ModelNet40Dataset(train_X, train_y, transform=test_transforms)
    test_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)

    extract()