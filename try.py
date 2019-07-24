from data_manager.data_process import get_while_running, kill_data_processes
from data_manager.shapenet import ShapenetDataProcess
from data_manager.vis import plot_pcds
from utils.train_utils import create_optimizer, train
from utils.parse_args import parse_args
from multiprocessing import Queue
from models.encoders import PointNetfeat
import torch


# Getting the arguments
args = parse_args()
# Loading the model
args.model = PointNetfeat(args)

# Testing for a GPU
args.gpu = torch.cuda.is_available()

# Loading the data
data_processes = []
data_queue = Queue(1)

for i in range(args.nworkers):
    data_processes.append(ShapenetDataProcess(data_queue, args, split='train', repeat=False))
    data_processes[-1].start()

# Getting a single batch of data
targets, clouds_data = data_queue.get()
# targets = torch.from_numpy(targets)
clouds_data = torch.from_numpy(clouds_data[1])

x = args.model.forward(clouds_data)
print("Shape of output is {}".format(x.shape))

# Defining the optimizer
args.optimizer = create_optimizer(args, args.model)

args.step = eval(args.NET + '_step')

# output = args.step(args, targets, clouds_data)
train(args, data_queue, data_processes)

kill_data_processes(data_queue, data_processes)
