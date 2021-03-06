from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from dataset import IDDataset, collate_fn, vocab, idx_to_word
from net import CNN, LSTM
from torch.autograd import Variable
from torch.utils.data import DataLoader

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # load data set
    is_training = True
    training_data = IDDataset(is_training);
    testing_data = IDDataset(not is_training)


    # Build data loader
    data_loader = DataLoader(training_data, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers,
                             collate_fn=collate_fn)

    # Build the models
    encoder = CNN(args.hidden_size)
    decoder = LSTM(args.embed_size, args.hidden_size,
                         len(vocab), args.num_layers)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Train the Models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, (image_batch, id_batch) in enumerate(data_loader):
            # Set mini-batch dataset
            images = to_var(image_batch)
            captions = to_var(id_batch)
            targets = to_var(id_batch[:,1:])

            # Forward, Backward and Optimize
            decoder.zero_grad()
            encoder.zero_grad()
            features = encoder(images)
            outputs = decoder(features, captions)
            loss = 0
            id_len = targets.size()[1]
            for j in xrange(id_len):
                loss += criterion(outputs[:,j,:], targets[:,j]) / id_len
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                      % (epoch, args.num_epochs, i, total_step,
                         loss.cpu().data.numpy()))

    # Save the models
    torch.save(decoder.state_dict(),
               os.path.join(args.model_path,
                            'decoder-%d-%d.pkl' % (epoch + 1, i + 1)))
    torch.save(encoder.state_dict(),
               os.path.join(args.model_path,
                            'encoder-%d-%d.pkl' % (epoch + 1, i + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./model/',
                        help='path for saving trained models')
    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for printing log info')
    parser.add_argument('--save_step', type=int, default=1000,
                        help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=16,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
