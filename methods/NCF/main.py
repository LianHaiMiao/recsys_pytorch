# please download the dataset from https://github.com/hexiangnan/neural_collaborative_filtering first.
# then, add the data files in data directory
# It worth to mention that we do not implement the pre-train version.
# we provide two kind of model: MLP and GMF

import time
import argparse

from dataset import NCFDataset
from model import NCF
import helper

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data



def parser_hyper_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="learning rate")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.0,
                        help="dropout rate")
    parser.add_argument("--batch_size",
                        type=int,
                        default=256,
                        help="batch size for training")
    parser.add_argument("--epochs",
                        type=int,
                        default=20,
                        help="training epoches")
    parser.add_argument("--top_k",
                        type=int,
                        default=10,
                        help="compute metrics@top_k")
    parser.add_argument("--embed_size",
                        type=int,
                        default=32,
                        help="predictive factors numbers in the model")
    parser.add_argument("--num_layers",
                        type=int,
                        default=3,
                        help="number of layers in MLP model")
    parser.add_argument("--num_ng",
                        type=int,
                        default=4,
                        help="sample negative items for training")
    parser.add_argument("--test_num_ng",
                        type=int,
                        default=99,
                        help="sample part of negative items for testing")
    parser.add_argument("--dataset",
                        type=str,
                        default="ml-1m",
                        help="the dataset for train and test")
    parser.add_argument("--model",
                        type=str,
                        default="NeuMF",
                        help="choose the model you need to train")
    parser.add_argument("--gpu",
                        type=str,
                        default="True",
                        help="whether utilize GPU")
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    # get hyper parameters
    args = parser_hyper_parameters()

    #####  data part #####
    ncf_dataset = NCFDataset(args.dataset)
    train_data, test_data, user_num, item_num, train_mat = ncf_dataset.load_dataset()
    # get train dataLoader
    train_data_loader = ncf_dataset.get_train_dataloader(
        args.batch_size, train_data, item_num, args.num_ng, train_mat)
    # get test dataLoader
    test_data_loader = ncf_dataset.get_test_dataloader(test_data, (args.test_num_ng+1))
    print("data prepare is over!")

    #####  model part #####
    ncf_model = NCF(user_num, item_num, args.embed_size, args.num_layers, args.dropout, args.model)
    if args.gpu:
        ncf_model.cuda()

    # loss function
    loss_function = nn.BCEWithLogitsLoss()
    # optimizer
    optimizer = optim.Adam(ncf_model.parameters(), lr=args.lr)

    #####  train and test part #####
    best_hr = 0
    for epoch in range(args.epochs):
        # begin to train
        ncf_model.train()
        begin_time = time.time()

        for user, item, label in train_data_loader:

            if args.gpu:
                user = user.cuda()
                item = item.cuda()
                label = label.float().cuda()

            ncf_model.zero_grad()
            prediction = ncf_model(user, item)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()

        # begin to test
        ncf_model.eval()
        HR, NDCG = helper.metrics(ncf_model, test_data_loader, args.gpu, args.top_k)

        print('Iteration %d consumes time [%.1f s]: HR = %.4f, NDCG = %.4f' % (
            epoch, time.time() - begin_time, HR, NDCG))

