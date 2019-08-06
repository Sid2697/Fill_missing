"""
This file contains the code similar to the implementation of the code in paper Deep Image Prior
"""

import torch
from models.PointNetFCAE import *
from models.MLP import *
from utils.train_utils import chamfer
from multiprocessing import Queue
from utils.parse_args import parse_args
from utils.train_utils import create_optimizer, train
from data_manager.shapenet import ShapenetDataProcess
from data_manager.data_process import kill_data_processes

epoch = 200

args = parse_args()
# args.model = PointNetFCAE_create_model(args)
args.model = MLP()

data_processes = []
data_queue = Queue(1)

for i in range(args.nworkers):
    data_processes.append(ShapenetDataProcess(data_queue, args, split='train', repeat=False))
    data_processes[-1].start()

# args.error = torch.nn.MSELoss()
# args.error = ChamferLoss()
args.optimizer = create_optimizer(args, args.model)

i = 0

while i != epoch:
    train(args, data_queue, data_processes, i)
    i += 1

kill_data_processes(data_queue, data_processes)
