from models.encoders import PointNetfeat, weight_init
from torch.autograd import Variable
from utils.distance import chamfer_distance_sklearn
import torch.nn.functional as F
import torch.nn as nn
import torch


def AtlasNet_step(args, targets_in, clouds_data):
    # TODO: This method is not yet complete
    targets = Variable(torch.from_numpy(targets_in), requires_grad=False).float()
    targets = targets.transpose(2, 1).contiguous()
    inp = Variable(torch.from_numpy(clouds_data[1]), requires_grad=False).float()
    output = args.model.forward(inp, args.grid)
    dist = chamfer_distance_sklearn(targets.numpy(), inp.numpy())
    return dist


class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=2500):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = nn.Conv1d(self.bottleneck_size, int(self.bottleneck_size/2), 1)
        self.conv3 = nn.Conv1d(int(self.bottleneck_size/2), int(self.bottleneck_size/4), 1)
        self.conv4 = nn.Conv1d(int(self.bottleneck_size/4), 3, 1)

        self.th = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = nn.BatchNorm1d(int(self.bottleneck_size / 2))
        self.bn3 = nn.BatchNorm1d(int(self.bottleneck_size / 4))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x


class AtlasNet(nn.Module):
    def __init__(self, args, num_points=2048, bottleneck_size=1024, nb_primitives=1):
        super(AtlasNet, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = nn.Sequential(
            PointNetfeat(args, num_points, global_feat=True, trans=False),
            nn.Linear(1024, self.bottleneck_size),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU()
        )
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size=2 + self.bottleneck_size) for i in range(0, self.nb_primitives)])

    def forward(self, x, grid):
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0, 1).contiguous.unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0), rand_grid.size(1), rand_grid.size(2)).contiguous()
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2, 1).contiguous()
