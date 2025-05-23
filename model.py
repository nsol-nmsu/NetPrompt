import torch
from torch import nn

class MLP_Mult(nn.Module):
    def __init__(self, input_shape, first_hidden, second_hidden, num_units, num_classes):
        super(MLP_Mult, self).__init__()
        self.fc1 = nn.Linear(input_shape, first_hidden)
        self.fc2 = nn.Linear(first_hidden, second_hidden)
        self.fc3 = nn.Linear(second_hidden, num_units)
        self.fc4 = nn.Linear(num_units, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x   