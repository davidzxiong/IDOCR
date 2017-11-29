from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
import torch
from dataset import vocab
from torch.autograd import Variable

class CNN(nn.Module):
    def __init__(self, hidden_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.linear = nn.Linear(20 * 53 * 53, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, image):
        x = F.relu(F.max_pool2d(self.conv1(image), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 20 * 53 * 53)
        features = self.bn(self.linear(x))
        return features


class LSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(LSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions[:,:-1])
        batch_size = features.size()[0]
        cell_state = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda())
        hidden_state = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda())
        for i in xrange(self.num_layers):
            hidden_state[i, :, :] = features
        output, _ = self.lstm(embeddings, (hidden_state, cell_state))
        output = self.linear(output)
        return output

    def sample(self, features, start):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        cell_state = Variable(torch.zeros(self.num_layers, 1, self.hidden_size).cuda())
        hidden_state = Variable(torch.zeros(self.num_layers, 1, self.hidden_size).cuda())
        for i in xrange(self.num_layers):
            hidden_state[i, :, :] = features
        states = (hidden_state, cell_state)
        inputs = self.embed(start)
        for i in range(18):  # maximum sampling length
            output, states = self.lstm(inputs, states)  # (batch_size, 1, hidden_size)
            output = self.linear(output)  # (batch_size, vocab_size)
            predicted = output.max(2)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
        sampled_ids = torch.cat(sampled_ids, 1)  # (batch_size, 20)
        return sampled_ids.squeeze()