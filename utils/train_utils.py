# from losses import chamfer_distance_sklearn
from data_manager.vis import plot
import torch.optim as optim
import torchnet as tnt
from tqdm import tqdm
import torch


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
    partial_one = partial[1:3]

    for batch in tqdm(range(num_batches)):
        # Zeroing the previous grads
        args.optimizer.zero_grad()
        out, code = args.model(partial_one)
        out = out.transpose(2, 1).contiguous()
        loss = args.error(out, partial_one)
        loss.backward()
        if batch % 10 == 0:
            plot(batch, epoch, partial_one[1].detach().numpy(), out[1].detach().numpy())

        args.optimizer.step()
