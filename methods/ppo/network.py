import torch
import torch.nn as nn
import  torch.optim as optim



class Policy(nn.Module):
    def __init__(self, action_size, lr, k_epoch, input_size = 4):
        super(Policy, self).__init__()
        self.action_size = action_size
        self.input_size = input_size
        self.layer1 = nn.Linear(self.input_size, 24)
        self.layer2 = nn.Linear(24, 24)
        self.layer3 = nn.Linear(24, self.action_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = -1)
        self.loss = 0.0
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = k_epoch, gamma = 0.999)
        self.criterion = nn.MSELoss()

    
    def pi(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return self.softmax(x)



class Value(nn.Module):
    def __init__(self, action_size, lr, k_epoch, input_size = 4):
        super(Value, self).__init__()
        self.action_size = action_size
        self.input_size = input_size
        self.layer4 = nn.Linear(self.input_size, 24)
        self.layer5 = nn.Linear(24, 24)
        self.layer6 = nn.Linear(24, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = -1)
        self.loss = 0.0
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = k_epoch, gamma = 0.999)
        self.criterion = nn.MSELoss()
    
    def v(self, x):
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        x = self.layer6(x)
        return x


class Network(nn.Module):

    def __init__(self, action_size, lr, k_epoch, input_size = 4):
        super(Network, self).__init__()
        self.action_size = action_size
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3_pi = nn.Linear(24, self.action_size)
        self.fc3_v = nn.Linear(24, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.loss = 0.0
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = k_epoch, gamma = 0.999)
        self.criterion = nn.MSELoss()

    def pi(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3_pi(x)
        return self.softmax(x)

    def v(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3_v(x)
        return x


