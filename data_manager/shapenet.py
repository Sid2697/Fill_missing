"""
This piece of code was taken from https://github.com/lynetcha/completion3d/tree/master/shared
"""
import os
import sys
import argparse
import numpy as np
# from vis import plot_pcds
from .data_process import DataProcess, get_while_running, kill_data_processes
from .data_utils import load_h5, load_csv, augment_cloud, pad_cloudN
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class ShapenetDataProcess(DataProcess):

    def __init__(self, data_queue, args, split='train', repeat=True):
        """
        Shapenet dataloader.

        Args:
            data_queue: multiprocessing queue where data is stored at.
            split: str in ('train', 'val', 'test'). Loads corresponding dataset.
            repeat: repeats epoch if true. Terminates after one epoch otherwise.
        """
        self.args = args
        self.split = split
        args.DATA_PATH = '/Volumes/Storage/Documents/IIT/Fill_missing/shapenet'
        classmap = load_csv(args.DATA_PATH + '/synsetoffset2category.txt')
        """
        This dictionary links class number with class names.
        """
        args.classmap = {}
        for i in range(classmap.shape[0]):
            args.classmap[str(classmap[i][1]).zfill(8)] = classmap[i][0]
        self.data_paths = sorted([os.path.join(args.DATA_PATH, split,
                                               'partial', k.rstrip() +
                                               '.h5') for k in open(args.DATA_PATH + '/%s.list' % (split)).readlines()])
        """
        Number of examples to use by making N divisible by 32.
        """
        N = int(len(self.data_paths)/args.batch_size)*args.batch_size
        """
        Selecting only first N point clouds
        """
        self.data_paths = self.data_paths[0:N]
        super().__init__(data_queue, self.data_paths, None, args.batch_size, repeat=repeat)

    def get_pair(self, args, fname, train):
        """
        This method is used to get partial and ground truth pair.
        :param args: Arguments
        :param fname: Name of the file
        :param train: Flag to indicate status of training
        :return: Partial and ground truth point clouds
        """
        partial = load_h5(fname)
        if self.split == 'test':
            gtpts = partial
        else:
            gtpts = load_h5(fname.replace('partial', 'gt'))
        if train:
            gtpts, partial = augment_cloud([gtpts, partial], args)
        # TODO[DONE]: Check the input and output shape in pad_cloudN
        partial = pad_cloudN(partial, args.inpts)
        return partial, gtpts

    def load_data(self, fname):
        pair = self.get_pair(self.args, fname, train=self.split == 'train')
        partial = pair[0].T
        target = pair[1]
        cloud_meta = ['{}.{:d}'.format('/'.join(fname.split('/')[-2:]), 0), ]
        return target[np.newaxis, ...], cloud_meta, partial[np.newaxis, ...]

    def collate(self, batch):
        targets, clouds_meta, clouds = list(zip(*batch))
        targets = np.concatenate(targets, 0)
        if len(clouds_meta[0])>0:
            clouds = np.concatenate(clouds, 0)
            clouds_meta = [item for sublist in clouds_meta for item in sublist]
        return targets, (clouds_meta, clouds)


def test_process():
    from multiprocessing import Queue
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.dataset = 'shapenet'
    args.nworkers = 4
    args.batch_size = 32
    args.pc_augm_scale = 0
    args.pc_augm_rot = 0
    args.pc_augm_mirror_prob = 0
    args.pc_augm_jitter = 0
    args.inpts = 2048
    """
    Process class from multiprocessing is used to run a piece of code separately from the parent code i.e. on a
    different core. Two important function of this class are start() and join(). start() runs the function on a 
    different core, and join() tells the independent process is complete.
    """
    data_processes = []
    """
    Queue is basically a First-In-First-Out data structure. Are useful for sharing between the process.
    They are useful when passed as a parameter to the Process' target function to enable the Process to consume data.
    put() and get() are the main function of the Queue class.
    """
    data_queue = Queue(1)
    for i in range(args.nworkers):
        data_processes.append(ShapenetDataProcess(data_queue, args, split='train', repeat=False))
        data_processes[-1].start()

    for targets, clouds_data in get_while_running(data_processes, data_queue, 0.5):
        """
        targets (Ground Truth) is a numpy.ndarray of shape (32, 2048, 3) i.e. it has 32 point clouds (1 batch) 
        of shape (2048, 3), clouds_data is a tuple of length 2, first element of tuple contains list of paths to 
        the point clouds, second element of the tuple contains numpy.ndarray is shape (32, 2048, 3). 
        In the following code, one sample of shape (2048, 3) is shown as partial input and ground truth. 
        """
        inp = clouds_data[1][0].squeeze().T
        targets = targets[0]
        # plot_pcds(None, [inp.squeeze(), targets.squeeze()], ['partial', 'gt'], use_color=[0, 0], color=[None, None])

    kill_data_processes(data_queue, data_processes)


if __name__ == '__main__':
    test_process()
