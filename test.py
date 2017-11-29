from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from torch.autograd import Variable
from torchvision import transforms
from net import CNN, LSTM
from dataset import IDDataset, vocab, idx_to_word
from PIL import Image


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def evaluation(data, encoder, decoder):
    correct_count = 0
    for i in xrange(len(data)):
        image, id = data[i]
        image_tensor = to_var(image.unsqueeze(0))
        # Generate caption from image
        feature = encoder(image_tensor)
        start = torch.LongTensor([vocab['<start>']]).unsqueeze(0)
        start = to_var(start)
        sampled_ids = decoder.sample(feature, start)
        sampled_ids = sampled_ids.cpu().data.numpy()

        # Decode word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = idx_to_word[word_id]
            sampled_caption.append(word)
        sentence = ''.join(sampled_caption)
        # Print out image and generated caption.
        targets = id.long().numpy()
        for i in xrange(18):
            predict = int(sampled_ids[i])
            target = int(targets[i+1])
            if predict == target:
                correct_count += 1
    return correct_count / 18 / len(data)


def main(args):

    # Build Models
    encoder = CNN(args.hidden_size)
    encoder.eval()  # evaluation mode (BN uses moving mean/variance)
    decoder = LSTM(args.embed_size, args.hidden_size,
                         len(vocab), args.num_layers)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # load data set
    is_training = True
    testing_data = IDDataset(not is_training)

    # If use gpu
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    test_acc = evaluation(testing_data, encoder, decoder)

    print("Accuracy is %.4f" % test_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, default='./model/encoder-100-133.pkl',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='./model/decoder-100-133.pkl',
                        help='path for trained decoder')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int, default=16,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in lstm')
    args = parser.parse_args()
    print(args)
    main(args)