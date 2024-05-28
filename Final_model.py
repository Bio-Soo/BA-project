# import part
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary

class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()
        self.features = \
            nn.Sequential(\
            # [1024,1,4,101] --> [1024,8,1,70]
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(32,4)),
            nn.ReLU(),
            # [1024,8,1,70] --> [1024,32,1,70] --> [1024,32,1,50]
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(21,1)),
            nn.ReLU(),
            # [1024,32,1,50] --> [1024,128,1,50] --> [1024,128,1,36]
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(15,1)),
            nn.ReLU(),
                )

        self.flatten = Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=4608, out_features=400),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=400, out_features=2))

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    x = torch.randn((1024,1,101,4))
    label = torch.randint(0,1,size=(1024,1))
    model = ConvolutionalNetwork()
    dataloader = DataLoader(torch.utils.data.TensorDataset(x, label),batch_size=1024, shuffle=True)
    out = model(x)
    summary(model, input_size=(1024,1,101,4))
    # print(out.shape)
    print(model)
