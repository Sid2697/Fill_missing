import torch.optim as optim
import torchnet as tnt
from tqdm import tqdm


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


def train(args, data_queue, data_processes):
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

    for batch in tqdm(range(num_batches)):
        # Getting data
        targets, clouds_data = data_queue.get()
        # Zeroing the previous grads
        args.optimizer.zero_grad()
        # TODO: Take up the code from here
