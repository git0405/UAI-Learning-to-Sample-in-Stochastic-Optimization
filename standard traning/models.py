import torch.nn as nn
import torch.nn.functional as F
import torch

import torchvision.models as models
class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU()
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class Siamese_ResNet(nn.Module):
    def __init__(self ):
        super(Siamese_ResNet, self).__init__()

        resnet18_model = models.resnet18(pretrained=True)
        self.resnet18_model = nn.Sequential(*list(resnet18_model.children())[:-1])
        self.fc1 = nn.Linear(512, 2)


    def forward_once(self, x):
        out = self.resnet18_model(x)

        out = out.view(out.size(0), -1)

        return out

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        dis = torch.square(output1 - output2)
        out = self.fc1(dis)
        output = F.log_softmax(out, dim=1)

        return output