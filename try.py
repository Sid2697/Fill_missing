from data_manager.data_process import get_while_running, kill_data_processes
from data_manager.shapenet import ShapenetDataProcess
from data_manager.vis import plot_pcds
from utils.train_utils import create_optimizer
from utils.parse_args import parse_args
from multiprocessing import Queue
from models.pcn import *
import torch


# Getting the arguments
args = parse_args()
# Loading the model
args.model = STN3d(args)

# Loading the data
data_processes = []
data_queue = Queue(1)

for i in range(args.nworkers):
    data_processes.append(ShapenetDataProcess(data_queue, args, split='train', repeat=False))
    data_processes[-1].start()

targets, clouds_data = data_queue.get()
targets = torch.from_numpy(targets)
clouds_data = torch.from_numpy(clouds_data[1])

x = args.model.forward(clouds_data)

# Defining the optimizer
args.optimizer = create_optimizer(args, args.model)

kill_data_processes(data_queue, data_processes)
