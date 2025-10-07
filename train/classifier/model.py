import torch.nn as nn

class SimpleClassifierNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleClassifierNN, self).__init__()
        self.layer_1 = nn.Linear(input_dim, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.dropout(x)
        x = self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.output_layer(x)
        return x