import torch
import torch.nn as nn
import torch.nn.init as init
import FFDNet_functions
from torch.autograd import Variable


class Conv_BN_Relu_first(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, groups, bias):
        super(Conv_BN_Relu_first, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups = 1
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                              groups=groups, bias=False)
        # self.conv = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv_BN_Relu_other(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, groups, bias):
        super(Conv_BN_Relu_other, self).__init__()
        kernel_size = 3
        padding = 1
        features = out_channels
        groups = 1
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, groups, bais):
        super(Conv, self).__init__()
        kernel_size = 3
        padding = 1
        features = 1
        groups = 1
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                              groups=groups, bias=False)

    def forward(self, x):
        return self.conv(x)


class Self_Attn(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        print(proj_query.size())
        print(proj_key.size())
        print('5')
        energy = torch.bmm(proj_query, proj_key)
        print('6')
        # print energy.size()
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        print('1')
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        print('2')
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x
        return out, attention


class ADNet(nn.Module):
    def __init__(self, channels, num_of_layers=15):
        super(ADNet, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups = 1
        layers = []
        kernel_size1 = 1
        '''
        #self.gamma = nn.Parameter(torch.zeros(1))
        '''
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_5 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_6 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_7 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_8 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_9 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_10 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_11 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_12 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_13 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_14 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_15 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_16 = nn.Conv2d(in_channels=features, out_channels=1, kernel_size=kernel_size, padding=1,
                                  groups=groups, bias=False)
        self.conv3 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.Tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)

    def _make_layers(self, block, features, kernel_size, num_of_layers, padding=1, groups=1, bias=False):
        layers = []
        for _ in range(num_of_layers):
            layers.append(block(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                                groups=groups, bias=bias))
        return nn.Sequential(*layers)

    def forward(self, x):
        input = x
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)
        x1 = self.conv1_5(x1)
        x1 = self.conv1_6(x1)
        x1 = self.conv1_7(x1)
        x1t = self.conv1_8(x1)
        x1 = self.conv1_9(x1t)
        x1 = self.conv1_10(x1)
        x1 = self.conv1_11(x1)
        x1 = self.conv1_12(x1)
        x1 = self.conv1_13(x1)
        x1 = self.conv1_14(x1)
        x1 = self.conv1_15(x1)
        x1 = self.conv1_16(x1)
        out = torch.cat([x, x1], 1)
        out = self.Tanh(out)
        out = self.conv3(out)
        out = out * x1
        out2 = x - out
        return out2


class ADNetres(nn.Module):
    def __init__(self, channels, num_of_layers=15):
        super(ADNetres, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups = 1
        layers = []
        kernel_size1 = 1
        '''
        #self.gamma = nn.Parameter(torch.zeros(1))
        '''
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_5 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_6 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_7 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_8 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_9 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_10 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_11 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_12 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_13 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_14 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_15 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_16 = nn.Conv2d(in_channels=features, out_channels=1, kernel_size=kernel_size, padding=1,
                                  groups=groups, bias=False)
        self.conv3 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.Tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)

    def _make_layers(self, block, features, kernel_size, num_of_layers, padding=1, groups=1, bias=False):
        layers = []
        for _ in range(num_of_layers):
            layers.append(block(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                                groups=groups, bias=bias))
        return nn.Sequential(*layers)

    def forward(self, x):
        input = x
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)
        x1 = self.conv1_5(x1)
        x1 = self.conv1_6(x1)
        x1 = self.conv1_7(x1)
        x1t = self.conv1_8(x1)
        x1 = self.conv1_9(x1t)
        x1 = self.conv1_10(x1)
        x1 = self.conv1_11(x1)
        x1 = self.conv1_12(x1)
        x1 = self.conv1_13(x1)
        x1 = self.conv1_14(x1)
        x1 = self.conv1_15(x1)
        x1 = self.conv1_16(x1)
        out = torch.cat([x, x1], 1)
        out = self.Tanh(out)
        out = self.conv3(out)
        out = out * x1
        # out2 = x - out
        return out


class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out


class DnCNN2(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN2, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(
            nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                          bias=False))
            # layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding,
                      bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out


class UpSampleFeatures(nn.Module):
    r"""Implements the last layer of FFDNet
    """
    def __init__(self):
        super(UpSampleFeatures, self).__init__()

    def forward(self, x):
        return FFDNet_functions.upsamplefeatures(x)


class IntermediateDnCNN(nn.Module):
    r"""Implements the middel part of the FFDNet architecture, which
    is basically a DnCNN net
    """
    def __init__(self, input_features, middle_features, num_conv_layers):
        super(IntermediateDnCNN, self).__init__()
        self.kernel_size = 3
        self.padding = 1
        self.input_features = input_features
        self.num_conv_layers = num_conv_layers
        self.middle_features = middle_features
        if self.input_features == 5:
            self.output_features = 4   # Grayscale image
        elif self.input_features == 15:
            self.output_features = 12  # RGB image
        else:
            raise Exception('Invalid number of input features')

        layers = []
        layers.append(nn.Conv2d(in_channels=self.input_features,\
                                out_channels=self.middle_features,\
                                kernel_size=self.kernel_size,\
                                padding=self.padding,\
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(self.num_conv_layers-2):
            layers.append(nn.Conv2d(in_channels=self.middle_features,\
                                    out_channels=self.middle_features,\
                                    kernel_size=self.kernel_size,\
                                    padding=self.padding,\
                                    bias=False))
            layers.append(nn.BatchNorm2d(self.middle_features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=self.middle_features,\
                                out_channels=self.output_features,\
                                kernel_size=self.kernel_size,\
                                padding=self.padding,\
                                bias=False))
        self.itermediate_dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.itermediate_dncnn(x)
        return out


class FFDNet(nn.Module):
    r"""Implements the FFDNet architecture
    """
    def __init__(self, num_input_channels):
        super(FFDNet, self).__init__()
        self.num_input_channels = num_input_channels
        if self.num_input_channels == 1:
            # Grayscale image
            self.num_feature_maps = 64
            self.num_conv_layers = 15
            self.downsampled_channels = 5
            self.output_features = 4
        elif self.num_input_channels == 3:
            # RGB image
            self.num_feature_maps = 96
            self.num_conv_layers = 12
            self.downsampled_channels = 15
            self.output_features = 12
        else:
            raise Exception('Invalid number of input features')

        self.intermediate_dncnn = IntermediateDnCNN(\
            input_features=self.downsampled_channels,\
            middle_features=self.num_feature_maps,\
            num_conv_layers=self.num_conv_layers)
        self.upsamplefeatures = UpSampleFeatures()

    def forward(self, x, noise_sigma):
        concat_noise_x = FFDNet_functions.concatenate_input_noise_map(\
            x.data, noise_sigma.data)
        concat_noise_x = Variable(concat_noise_x)
        h_dncnn = self.intermediate_dncnn(concat_noise_x)
        pred_noise = self.upsamplefeatures(h_dncnn)
        return pred_noise
