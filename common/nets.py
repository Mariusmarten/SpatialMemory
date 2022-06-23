# nets
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Forward(nn.Module):
    def __init__(self, output_neurons):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_neurons)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Forward_Large(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 10)
        self.fc1 = nn.Linear(16 * 5 * 10, 240)
        self.fc2 = nn.Linear(240, 168)
        self.fc3 = nn.Linear(168, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DualOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch

        # output1
        x0 = F.relu(self.fc1(x))
        x0 = F.relu(self.fc2(x0))
        x0 = self.fc3(x0)

        # output2
        x1 = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2(x1))
        x1 = self.fc3(x1)
        return x0, x1

class DualInput(nn.Module):
    def __init__(self, output_neurons):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5 * 2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_neurons)

    def forward(self, x0, x1):
        # input 0
        x0 = self.pool(F.relu(self.conv1(x0)))
        x0 = self.pool(F.relu(self.conv2(x0)))
        x0 = torch.flatten(x0, 1) # flatten all dimensions except batch

        # input 1
        x1 = self.pool(F.relu(self.conv1(x1)))
        x1 = self.pool(F.relu(self.conv2(x1)))
        x1 = torch.flatten(x1, 1) # flatten all dimensions except batch

        x = torch.cat((x0, x1), dim=1)
        #x = x0 + x1

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# original_version
class CNN_coords(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv3 = nn.Conv2d(20, 30, 5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, i):
        #x = i.view(-1, i.shape[2], i.shape[3], i.shape[4])
        x = i.reshape(-1, i.shape[2], i.shape[3], i.shape[4])
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(i.shape[0], i.shape[1], -1)
        return x

class LSTM_coords(nn.Module):
    def __init__(self, length_trajectory):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(480, 100, 2)
        self.fc = nn.Linear(100*10, length_trajectory)

    def forward(self, x, hn, cn):
        x, (hn, cn) = self.lstm(x, (hn, cn))
        hn, cn = (hn, cn)
        x = F.relu(x.view(x.shape[0], -1))

        output0 = F.relu(self.fc(x))
        output1 = F.relu(self.fc(x))
        # alternatively we could just return the final hidde
        return output0, output1, hn, cn

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc1 = nn.Linear(hidden_size, int(hidden_size*0.5))
        self.fc2 = nn.Linear(int(hidden_size*0.5), num_classes)

    def forward(self, x):
        # newly initialized only after each epoch
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = F.relu(self.fc1(h_out))
        out = self.fc2(out)

        return out

class CNN(nn.Module):

    def __init__(self, seq_length):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, seq_length, 5)
        self.conv2 = nn.Conv2d(seq_length, seq_length*2, 5)
        self.conv3 = nn.Conv2d(seq_length*2, seq_length*3, 5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, i):
        x = i.reshape(-1, i.shape[2], i.shape[3], i.shape[4]) # merges batch and length dimension
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(i.shape[0], i.shape[1], -1)

        return x
