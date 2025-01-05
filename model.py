import torch
import torch.nn as nn

class ChatBotNeuralNet(nn.Module):
    def __init__(self,n_input,n_hidden,n_classes):
        super().__init__()
        self.l1 = nn.Linear(n_input,n_hidden)
        self.l2 = nn.Linear(n_hidden,n_hidden)
        self.l3 = nn.Linear(n_hidden,n_classes)
        self.relu = nn.ReLU()

    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        
        return out

    
