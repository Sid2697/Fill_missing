"""
This file contains the code similar to the implementation of the code in paper Deep Image Prior
"""

import torch
import numpy as np
# from models.AtlasNet import *
from models.PointNetFCAE import *
from multiprocessing import Queue
from data_manager.vis import plot_pcds, plot_xyz
from utils.parse_args import parse_args
from data_manager.shapenet import ShapenetDataProcess
from data_manager.data_process import kill_data_processes

show = True

args = parse_args()
args.model = PointNetFCAE_create_model(args)

data_processes = []
data_queue = Queue(1)

for i in range(args.nworkers):
    data_processes.append(ShapenetDataProcess(data_queue, args, split='train', repeat=False))
    data_processes[-1].start()

targets, partial = data_queue.get()
partial = partial[1]
partial = torch.from_numpy(partial)
partial_plot = partial.transpose(2, 1).contiguous()

error = torch.nn.MSELoss()

out, code = args.model(partial)

if show:
    print("Plotting the point clouds!")
    plot_pcds(None, [out[0].detach().numpy(), partial_plot[0].detach().numpy()], ['Output', 'Input'],
              use_color=[0, 0], color=[None, None])

kill_data_processes(data_queue, data_processes)
