import torch
from torch import optim
import numpy as np
from torchvision import datasets, transforms
from models import resnet
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.nn import functional
from torch import nn
from utils.SimpleParser import SimpleDataset
from config.config import Config
# from tensorboardX import FileWriter
from tensorboardX import SummaryWriter

''' original batch_NN_loss used in my former work (simple_rnn/src/eval_ps_gen.py)
    author: Joey Guo
    
    def batch_NN_loss(x, y):
        bs, num_points, points_dim = x.size()
        xx = x.unsqueeze(-2).expand(bs, num_points, num_points, points_dim)
        yy = y.unsqueeze(-2).expand(bs, num_points, num_points, points_dim)
    
        D = xx - yy.transpose(-2, -3)
        D = torch.sqrt(torch.sum(D * D, dim=-1)) # sum over space dimension
        min_dist1, _ = torch.min(D, dim=-1)
        min_dist2, _ = torch.min(D, dim=-2)
        
        avg_emd = 0.5 * (min_dist1 + min_dist2).mean() # actually it is only approx. avg_emd
        return avg_emd

'''

""" The original version
Author: Joey Guo
For test purpose.
"""


def original_batch_NN_loss(x, y):
    bs, num_points, points_dim = x.size()
    xx = x.unsqueeze(-2).expand(bs, num_points, num_points, points_dim)
    yy = y.unsqueeze(-2).expand(bs, num_points, num_points, points_dim)

    D = xx - yy.transpose(-2, -3)
    D = torch.sqrt(torch.sum(D * D, dim=-1))  # sum over space dimension
    min_dist1, _ = torch.min(D, dim=-1)
    min_dist2, _ = torch.min(D, dim=-2)

    avg_emd = 0.5 * (min_dist1 + min_dist2).mean()  # actually it is only approx. avg_emd
    return avg_emd


""" new version when x does not share the same length (num of points) with y
Author: Joey Guo
input: x, of size (batchsize, num_of_points_x, dimensions)
input: y, of size (batchsize, num_of_points_y, dimensions)
return: the approx. avg emd

Example:
    x_npy = np.array([0.0, 1, 2])[None, :, None]
    y_npy = np.array([0.0, 1, 2, 3, 4])[None, :, None]
    # x = torch.randn(32, 1000, 1).cuda()
    # y = torch.randn(32, 1000, 1).cuda()
    x = torch.from_numpy(x_npy)
    y = torch.from_numpy(y_npy)

    print(batch_NN_loss(x, y))  # 0.3
"""


def batch_NN_loss(x, y):
    bs, num_points_x, points_dim = x.size()
    bs, num_points_y, points_dim = y.size()
    xx = x.unsqueeze(2).expand(bs, num_points_x, num_points_y, points_dim)  # x coord is fixed on dim 2 (start from 0)
    yy = y.unsqueeze(1).expand(bs, num_points_x, num_points_y, points_dim)  # y coord is fixed on dim 1 (start from 0)

    D = xx - yy
    D = torch.sqrt(torch.sum(D * D, dim=-1))  # sum over space dimension

    # fix x to search on ys, so this is the min dist from each point in x to the set of y
    min_dist1, _ = torch.min(D, dim=2)
    # fix y to search on xs, so this is the min dist from each point in y to the set of x
    min_dist2, _ = torch.min(D, dim=1)

    # actually it is only approx. avg_emd
    avg_emd = 0.5 * (min_dist1.mean() + min_dist2.mean())
    # avg_emd = (min_dist1.sum() + min_dist2.sum()) / (bs * (num_points_x + num_points_y))
    return avg_emd


def to_var(someTensor, is_cuda=True):
    retVar = Variable(someTensor)
    if is_cuda:
        return retVar.cuda()
    else:
        return retVar


class RPN(nn.Module):

    # numprops = num of proposals
    def __init__(self, inChan, numprops):
        super(RPN, self).__init__()
        self.conv1 = nn.Conv2d(inChan, inChan, 3, bias=False, padding=1)
        self.conv2 = nn.Conv2d(inChan, inChan, 3, bias=False, padding=1)
        self.convCoord = nn.Conv2d(inChan, numprops, 3, bias=False, padding=1)
        # self.convScore = nn.Conv2d(inChan, 1, 3, bias=False, padding=1)

    def forward(self, feat):
        feat = functional.relu(self.conv1(feat), True)
        feat = functional.relu(self.conv2(feat), True)
        coord = self.convCoord(feat)
        coord = functional.sigmoid(functional.adaptive_avg_pool2d(coord, (4, 1))).squeeze(-1)

        # score = functional.sigmoid(self.convScore(feat))

        return coord


class REG(nn.Module):
    def __init__(self, inChan):
        super(REG, self).__init__()
        self.conv1 = nn.Conv2d(inChan, inChan, 3, bias=False, padding=1)
        self.conv2 = nn.Conv2d(inChan, inChan, 3, bias=False, padding=1)
        self.convCoord = nn.Conv2d(inChan, 4, 3, bias=False, padding=1)
        self.convScore = nn.Conv2d(inChan, 1, 3, bias=False, padding=1)

    def forward(self, feat):
        feat = functional.relu(self.conv1(feat), True)
        feat = functional.relu(self.conv2(feat), True)
        coord = self.convCoord(feat)
        score = functional.sigmoid(self.convScore(feat))

        return coord, score


class DetNet(nn.Module):
    def __init__(self, lr=1e-4):
        super(DetNet, self).__init__()
        self.lr = lr
        self.feat = resnet.resnet34(True)
        self.rpn = self.build_rpn(512, 100)
        # self.reg = self.build_reg()
        # self.roi_pooling = nn.AdaptiveAvgPool2d((7, 7))
        self.opt = optim.Adam(self.parameters(), lr=self.lr)

    def build_rpn(self, inCh, outCh):
        # rpn = nn.
        rpn = RPN(inCh, outCh)
        return rpn

    def forward(self, x):
        feat = self.feat(x)
        proposals = self.rpn(feat)
        return proposals

    def step(self, xs, ys):
        var_xs, var_ys = to_var(xs), to_var(ys)
        var_props = net(var_xs)
        self.zero_grad()
        var_loss = batch_NN_loss(var_props, var_ys)
        var_loss.backward()
        self.opt.step()
        return var_props, var_loss


# write a stupid and dirty localizer
if __name__ == '__main__':
    _c = Config()
    net = DetNet()
    if _c.is_cuda:
        net.cuda()
    writer = SummaryWriter('../logs/exp01')
    ds = SimpleDataset(_c.imgroot, _c.xmlroot)
    dl = DataLoader(ds, 1, True)
    for epoch in range(1000):
        dataiter = iter(dl)

        for ii, minibatch in enumerate(dataiter, 0):
            xs, ys = minibatch
            # print(xs)  #
            # print(len(ys))
            # print(ys.size())  #
            # var_xs, var_ys = to_var(xs), to_var(ys)
            # var_props = net(var_xs)
            # var_loss = batch_NN_loss(var_props, var_ys)

            print(ii)
            var_props, var_loss = net.step(xs, ys)
            writer.add_scalar('loss', var_loss.data, ii + epoch * len(dl))
            print(var_loss)

            # print('var_props', var_props.size())
            # print('ys', ys.size())
            # print(var_loss)
