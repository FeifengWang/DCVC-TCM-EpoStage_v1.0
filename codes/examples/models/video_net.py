import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from model.subnet.endecoder import bilinearupsacling2

Backward_tensorGrid = [{} for i in range(8)]
Backward_tensorGrid_cpu = {}


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    def __init__(self,
                 ch,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=0.1,
                 reparam_offset=2**-18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset

        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset**2
        self.beta_bound = ((self.beta_min + self.reparam_offset**2)**0.5)
        self.gamma_bound = self.reparam_offset

        beta = torch.sqrt(torch.ones(ch)+self.pedestal)
        self.beta = nn.Parameter(beta)

        eye = torch.eye(ch)
        g = self.gamma_init*eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma)
        self.pedestal = self.pedestal

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d*w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs


def torch_warp(tensorInput, tensorFlow):
    if tensorInput.device == torch.device('cpu'):
        if str(tensorFlow.size()) not in Backward_tensorGrid_cpu:
            tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(
                1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
            tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(
                1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
            Backward_tensorGrid_cpu[str(tensorFlow.size())] = torch.cat(
                [tensorHorizontal, tensorVertical], 1).cpu()

        tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
                                tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)

        grid = (Backward_tensorGrid_cpu[str(tensorFlow.size())] + tensorFlow)
        return torch.nn.functional.grid_sample(input=tensorInput,
                                               grid=grid.permute(0, 2, 3, 1),
                                               mode='bilinear',
                                               padding_mode='border',
                                               align_corners=True)
    else:
        device_id = tensorInput.device.index
        if str(tensorFlow.size()) not in Backward_tensorGrid[device_id]:
            tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(
                1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
            tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(
                1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
            Backward_tensorGrid[device_id][str(tensorFlow.size())] = torch.cat(
                [tensorHorizontal, tensorVertical], 1).cuda().to(device_id)

        tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
                                tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)

        grid = (Backward_tensorGrid[device_id][str(tensorFlow.size())] + tensorFlow)
        return torch.nn.functional.grid_sample(input=tensorInput,
                                               grid=grid.permute(0, 2, 3, 1),
                                               mode='bilinear',
                                               padding_mode='border',
                                               align_corners=True)


def flow_warp(im, flow):
    warp = torch_warp(im, flow)
    return warp


def load_weight_form_np(me_model_dir, layername):
    index = layername.find('modelL')
    if index == -1:
        print('load models error!!')
    else:
        name = layername[index:index + 11]
        modelweight = me_model_dir + name + '-weight.npy'
        modelbias = me_model_dir + name + '-bias.npy'
        weightnp = np.load(modelweight)
        biasnp = np.load(modelbias)
        return torch.from_numpy(weightnp), torch.from_numpy(biasnp)


def bilinearupsacling(inputfeature):
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    outfeature = F.interpolate(
        inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear', align_corners=False)
    return outfeature


class ResBlock(nn.Module):
    def __init__(self, inputchannel, outputchannel, kernel_size, stride=1):
        super(ResBlock, self).__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(inputchannel, outputchannel,
                               kernel_size, stride, padding=kernel_size//2)
        torch.nn.init.xavier_uniform_(self.conv1.weight.data)
        torch.nn.init.constant_(self.conv1.bias.data, 0.0)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(outputchannel, outputchannel,
                               kernel_size, stride, padding=kernel_size//2)
        torch.nn.init.xavier_uniform_(self.conv2.weight.data)
        torch.nn.init.constant_(self.conv2.bias.data, 0.0)
        if inputchannel != outputchannel:
            self.adapt_conv = nn.Conv2d(inputchannel, outputchannel, 1)
            torch.nn.init.xavier_uniform_(self.adapt_conv.weight.data)
            torch.nn.init.constant_(self.adapt_conv.bias.data, 0.0)
        else:
            self.adapt_conv = None

    def forward(self, x):
        x_1 = self.relu1(x)
        firstlayer = self.conv1(x_1)
        firstlayer = self.relu2(firstlayer)
        seclayer = self.conv2(firstlayer)
        if self.adapt_conv is None:
            return x + seclayer
        else:
            return self.adapt_conv(x) + seclayer


class ResBlock_LeakyReLU_0_Point_1(nn.Module):
    def __init__(self, d_model):
        super(ResBlock_LeakyReLU_0_Point_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(d_model, d_model, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        x = x+self.conv(x)
        return x


class MEBasic(nn.Module):
    def __init__(self):
        super(MEBasic, self).__init__()
        self.conv1 = nn.Conv2d(8, 32, 7, 1, padding=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 7, 1, padding=3)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 32, 7, 1, padding=3)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 16, 7, 1, padding=3)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(16, 2, 7, 1, padding=3)


    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.conv5(x)
        return x


class ME_Spynet(nn.Module):
    def __init__(self):
        super(ME_Spynet, self).__init__()
        self.L = 4
        self.moduleBasic = torch.nn.ModuleList(
            [MEBasic() for intLevel in range(4)])

    def forward(self, im1, im2):
        batchsize = im1.size()[0]
        im1_pre = im1
        im2_pre = im2

        im1list = [im1_pre]
        im2list = [im2_pre]
        for intLevel in range(self.L - 1):
            im1list.append(F.avg_pool2d(
                im1list[intLevel], kernel_size=2, stride=2))
            im2list.append(F.avg_pool2d(
                im2list[intLevel], kernel_size=2, stride=2))

        shape_fine = im2list[self.L - 1].size()
        zeroshape = [batchsize, 2, shape_fine[2] // 2, shape_fine[3] // 2]
        device = im1.device
        flowfileds = torch.zeros(
            zeroshape, dtype=torch.float32, device=device)
        for intLevel in range(self.L):
            flowfiledsUpsample = bilinearupsacling(flowfileds) * 2.0
            flowfileds = flowfiledsUpsample + \
                self.moduleBasic[intLevel](torch.cat([im1list[self.L - 1 - intLevel],
                                                      flow_warp(im2list[self.L - 1 - intLevel],
                                                                flowfiledsUpsample),
                                                      flowfiledsUpsample], 1))

        return flowfileds
class Warp_net(nn.Module):
    def __init__(self):
        super(Warp_net, self).__init__()
        channelnum = 64

        self.feature_ext = nn.Conv2d(6, channelnum, 3, padding=1)# feature_ext
        self.f_relu = nn.ReLU()
        torch.nn.init.xavier_uniform_(self.feature_ext.weight.data)
        torch.nn.init.constant_(self.feature_ext.bias.data, 0.0)
        self.conv0 = ResBlock(channelnum, channelnum, 3)#c0
        self.conv0_p = nn.AvgPool2d(2, 2)# c0p
        self.conv1 = ResBlock(channelnum, channelnum, 3)#c1
        self.conv1_p = nn.AvgPool2d(2, 2)# c1p
        self.conv2 = ResBlock(channelnum, channelnum, 3)# c2
        self.conv3 = ResBlock(channelnum, channelnum, 3)# c3
        self.conv4 = ResBlock(channelnum, channelnum, 3)# c4
        self.conv5 = ResBlock(channelnum, channelnum, 3)# c5
        self.conv6 = nn.Conv2d(channelnum, 3, 3, padding=1)
        torch.nn.init.xavier_uniform_(self.conv6.weight.data)
        torch.nn.init.constant_(self.conv6.bias.data, 0.0)

    def forward(self, x):
        feature_ext = self.f_relu(self.feature_ext(x))
        c0 = self.conv0(feature_ext)
        c0_p = self.conv0_p(c0)
        c1 = self.conv1(c0_p)
        c1_p = self.conv1_p(c1)
        c2 = self.conv2(c1_p)
        c3 = self.conv3(c2)
        c3_u = c1 + bilinearupsacling2(c3)#torch.nn.functional.interpolate(input=c3, scale_factor=2, mode='bilinear', align_corners=True)
        c4 = self.conv4(c3_u)
        c4_u = c0 + bilinearupsacling2(c4)# torch.nn.functional.interpolate(input=c4, scale_factor=2, mode='bilinear', align_corners=True)
        c5 = self.conv5(c4_u)
        res = self.conv6(c5)
        return res