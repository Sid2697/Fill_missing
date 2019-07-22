import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Point cloud completion by Siddhant')

    parser.add_argument('--optim', default='adagrad', help='Optimizer: sgd|adam|adagrad|adadelta|rmsprop')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--wd', default=0, type=float, help='Weight Decay')
    parser.add_argument('--lr', default=1e-3, type=float, help='Initial Learning Rate')

    parser.add_argument('--NET', default='STN3d', help='Network used')
    parser.add_argument('--code_nfts', default=1024, type=int, help='Encoder output feature size')

    parser.add_argument('--batch_size', default=32, type=int, help='Batch Size')
    parser.add_argument('--nworkers', default=4, type=int, help='Number of subprocess for data loading')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epoch to train')

    # Arguments for dataset
    parser.add_argument('--dataset', default='shapenet', help='Dataset name: shapenet')
    parser.add_argument('--npts', default=2048, type=int, help='Number of output points generated')
    parser.add_argument('--inpts', default=2048, type=int, help='Number of input points')
    parser.add_argument('--ngtpts', default=2048, type=int, help='Number of ground-truth points')
    parser.add_argument('--pc_augm_scale', default=0, type=float, help='Training augmentation: Uniformly random '
                                                                       'scaling in [1/scale, scale]')
    parser.add_argument('--pc_augm_rot', default=0, type=int, help='Training augmentation: Bool, random rotation'
                                                                   'around z-axis')
    parser.add_argument('--pc_augm_mirror_prob', default=0, type=float,help='Training augmentation: Probability of '
                                                                            'mirroring about x or y axes')
    parser.add_argument('--pc_augm_jitter', default=0, type=int, help='Training augmentation: Bool, Gaussian'
                                                                      'jittering of all attributes')

    args = parser.parse_args()
    args.start_epoch = 0

    return args
