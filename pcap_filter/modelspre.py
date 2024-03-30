
import torch
import torch.nn.functional as F

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 4 * 4, 27)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc1(x)
        return x

class Cifar10CnnNet(torch.nn.Module):
    def __init__(self):
        super(Cifar10CnnNet, self).__init__()
        self.Conv = torch.nn.Sequential(

            torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(3, 3), stride=1, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=1, padding=2), torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(3, 3), stride=2)

        )

        self.linear = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 3 * 3, 1024), torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 1024), torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.Conv(x)
        x = self.linear(x)
        return x

class NSLKDDTestNet(torch.nn.Module):
    def __init__(self):
        super(NSLKDDTestNet, self).__init__()
        '''
        self.network = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, kernel_size=3, out_channels=32),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=32, kernel_size=3, out_channels=64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),  # output: 64 x 16 x 16
        )
        self.linear = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(2*64*59, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(128, 5),
            torch.nn.Softmax(),
        )   
        '''
        self.conv1 = torch.nn.Conv1d(in_channels=1, kernel_size=1, out_channels=32)
        self.conv2 = torch.nn.Conv1d(in_channels=32, kernel_size=2, out_channels=64)
        self.pool = torch.nn.AvgPool1d(2)

        self.linear = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 30, 128),
            torch.nn.Dropout(p=0.05),
            torch.nn.Linear(128, 5),
        )

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = self.linear(x)
        return x

class Vgg16_Net(torch.nn.Module):
    def __init__(self):
        super(Vgg16_Net, self).__init__()
        # 2个卷积层和1个最大池化层
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # (32-3+2)/1+1 = 32  32*32*64
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # (32-3+2)/1+1 = 32  32*32*64
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),

            torch.nn.MaxPool2d(2, 2)  # (32-2)/2+1 = 16    16*16*64

        )
        # 2个卷积层和1个最大池化层
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # (16-3+2)/1+1 = 16  16*16*128
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # (16-3+2)/1+1 = 16  16*16*128
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),

            torch.nn.MaxPool2d(2, 2)  # (16-2)/2+1 = 8    8*8*128
        )
        # 3个卷积层和1个最大池化层
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1 = 8  8*8*256
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1 = 8  8*8*256
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1 = 8  8*8*256
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),

            torch.nn.MaxPool2d(2, 2),  # (8-2)/2+1 = 4    4*4*256
        )
        # 3个卷积层和1个最大池化层
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # (4-3+2)/1+1 = 4  4*4*512
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # (4-3+2)/1+1 = 4  4*4*512
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # (4-3+2)/1+1 = 4  4*4*512
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),

            torch.nn.MaxPool2d(2, 2)  # (4-2)/2+1 = 2    2*2*512
        )
        # 3个卷积层和1个最大池化层
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # (2-3+2)/1+1 = 2  2*2*512
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # (2-3+2)/1+1 = 2  2*2*512
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # (2-3+2)/1+1 = 2  2*2*512
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),

            torch.nn.MaxPool2d(2, 2)  # (2-2)/2+1 = 1    1*1*512
        )
        self.conv = torch.nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),

            torch.nn.Linear(512, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),

            torch.nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x

class Vgg16_Net_forEMINIST(torch.nn.Module):
    def __init__(self):
        super(Vgg16_Net_forEMINIST, self).__init__()
        # 2个卷积层和1个最大池化层
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # (32-3+2)/1+1 = 32  32*32*64
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # (32-3+2)/1+1 = 32  32*32*64
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),

            torch.nn.MaxPool2d(2, 2)  # (32-2)/2+1 = 16    16*16*64

        )
        # 2个卷积层和1个最大池化层
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # (16-3+2)/1+1 = 16  16*16*128
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # (16-3+2)/1+1 = 16  16*16*128
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),

            torch.nn.MaxPool2d(2, 2)  # (16-2)/2+1 = 8    8*8*128
        )
        # 3个卷积层和1个最大池化层
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1 = 8  8*8*256
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1 = 8  8*8*256
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1 = 8  8*8*256
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),

            torch.nn.MaxPool2d(2, 2),  # (8-2)/2+1 = 4    4*4*256
        )
        # 3个卷积层和1个最大池化层
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # (4-3+2)/1+1 = 4  4*4*512
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # (4-3+2)/1+1 = 4  4*4*512
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # (4-3+2)/1+1 = 4  4*4*512
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),

            torch.nn.MaxPool2d(2, 2)  # (4-2)/2+1 = 2    2*2*512
        )
        # 3个卷积层和1个最大池化层
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # (2-3+2)/1+1 = 2  2*2*512
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # (2-3+2)/1+1 = 2  2*2*512
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # (2-3+2)/1+1 = 2  2*2*512
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),

            torch.nn.MaxPool2d(2, 2)  # (2-2)/2+1 = 1    1*1*512
        )
        self.conv = torch.nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),

            torch.nn.Linear(512, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),

            torch.nn.Linear(256, 26)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x

class EMINISTNET(torch.nn.Module):
    def __init__(self):
        super(EMINISTNET, self).__init__()
        self.block_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                      out_channels=64,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      # (1(32-1)- 32 + 3)/2 = 1
                      padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )

        self.block_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )

        self.block_3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )

        self.block_4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )

        self.block_5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 2 * 2, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 27)
        )
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, np.sqrt(2. / n))
                m.weight.detach().normal_(0, 0.05)
                if m.bias is not None:
                    m.bias.detach().zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.detach().normal_(0, 0.05)
                m.bias.detach().detach().zero_()

    def forward(self, x):

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)

        logits = self.classifier(x.view(-1, 512 * 2 * 2))
        #probas = F.softmax(logits, dim=1)

        return logits
