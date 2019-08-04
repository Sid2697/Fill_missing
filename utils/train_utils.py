# from losses import chamfer_distance_sklearn
from data_manager.vis import plot, plot_temp
import torch.optim as optim
import torchnet as tnt
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import torch


def add_noise(target):
    noise = np.random.uniform(-0.5, 0.5, (2, 3, 100))
    target = target[:, :, :1948]
    target = np.concatenate((noise, target), axis=2)
    return torch.from_numpy(target)


def create_optimizer(args, model):
    optimizer = None
    params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optim == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    elif args.optim == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(params, lr=args.lr)
    elif args.optim == 'adadelta':
        optimizer = optim.Adadelta(params, lr=args.lr, rho=0.9, epsilon=1e-6, weight_decay=args.wd)
    elif args.optim == 'rmsprop':
        optimizer = optim.RMSprop(params, lr=args.lr, alpha=0.99, epsilon=1e-8, weight_decay=args.wd)
    return optimizer


def train(args, data_queue, data_processes, epoch):
    """
    Training for one epoch
    :param args: Arguments for the code
    :param data_queue: Queue of data
    :param data_processes: Processes running data
    :return:
    """
    print("Training...")
    args.model.train()

    N = len(data_processes[0].data_paths)
    num_batches = int(N / args.batch_size)
    if num_batches * args.batch_size < N:
        num_batches += 1

    meters = list()
    lnm = ['loss']
    Nl = len(lnm)
    for i in range(Nl):
        meters.append(tnt.meter.AverageValueMeter())

    targets, partial = data_queue.get()
    partial = torch.from_numpy(partial[1])
    targets = torch.from_numpy(targets).float()
    # targets_one = targets[1:3]
    # targets = targets.transpose(2, 1).contiguous()
    # print("Shape of partial is {}, shape of target is {}".format(partial.shape, targets.shape))
    # input_noise = torch.from_numpy(np.random.uniform(-0.5, 0.5, (2, 3,
    #                                                  2048)))
    # noisy_target = add_noise(targets_one.numpy())

    for batch in tqdm(range(num_batches)):
        # Zeroing the previous grads
        args.optimizer.zero_grad()
        out = args.model(partial)
        # plot_temp(out[1].detach().numpy())
        # print("Shape of out is", out.shape)
        out_temp = out.transpose(2, 1).contiguous()
        # print('Out max is {}, out min is {}'.format(np.max(out.detach().numpy()), np.min(out.detach().numpy())))
        # print("Type out {}, type targets {}".format(out.type(), targets.type()))
        loss = args.error(out_temp, targets)
        loss.backward()
        # plot_temp(partial[1].detach().numpy())
        if batch % 100 == 0:
            print(loss)
            plot(batch, epoch, partial[1].detach().numpy(), out[1].detach().numpy())

        args.optimizer.step()
