# import part
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()
        self.features = \
            nn.Sequential(\
                # [1000,1,4,101] --> [1000,8,]
                nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(22,4)),
                nn.ReLU(),
                # [1000,8,21] --> [1000,16,17] --> [100,16,3]
                nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(15,1)),
                nn.ReLU(),
                # [1000,16,3] --> [1000,4,2] --> [1000,4,1]
                nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(15,1)),
                nn.ReLU(),
                )

        self.flatten = Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=6656, out_features=1),
            nn.Sigmoid(),
            nn.Dropout(p=0.9))

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    x = torch.randn((1000,1,101,4))
    label = torch.randint(0,1,size=(1000,1))
    model = ConvolutionalNetwork()
    dataloader = DataLoader(torch.utils.data.TensorDataset(x, label),batch_size=1000, shuffle=True)
    out = model(x)

    print(out.shape)
