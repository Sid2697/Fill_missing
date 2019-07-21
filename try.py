from data_manager.data_process import get_while_running, kill_data_processes
from data_manager.shapenet import ShapenetDataProcess
from utils.parse_args import parse_args
from multiprocessing import Queue
from models.pcn import *


# Getting the arguments
args = parse_args()
# Loading the model
model = STN3d(args)

# Loading the data
data_processes = []
data_queue = Queue(1)

for i in range(args.nworkers):
    data_processes.append(ShapenetDataProcess(data_queue, args, split='train', repeat=False))
    data_processes[-1].start()

for targets, clouds_data in get_while_running(data_processes, data_queue, 0.5):
    inp = clouds_data[1][0].squeeze().T
    targets = targets[0]
    print('Type of inp is {}, type of targets is {}'.format(type(inp), type(targets)))

kill_data_processes(data_queue, data_processes)
