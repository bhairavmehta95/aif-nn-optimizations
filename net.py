import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, l1=120, l2=84):
        super().__init__()
        self.linear1 = nn.Linear(4, l1)
        self.bn1 = nn.BatchNorm1d(l1)
        self.linear2 = nn.Linear(l1, l1)
        self.bn2 = nn.BatchNorm1d(l1)
        self.linear3 = nn.Linear(l1, l2)
        self.bn3 = nn.BatchNorm1d(l2)
        self.linear4 = nn.Linear(l2, l2)
        self.bn4 = nn.BatchNorm1d(l2)
        self.linear5 = nn.Linear(l2, 1)
    
    def forward(self, x):
        x = torch.tanh(self.bn1(self.linear1(x)))
        x = torch.tanh(self.bn2(self.linear2(x)))
        x = torch.tanh(self.bn3(self.linear3(x)))
        x = torch.tanh(self.bn4(self.linear4(x)))
        x = self.linear5(x)
        return x
        