import numpy as np

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as fn
import torch

from dataset import SeqDataset
from model2d import ConvolutionalNetwork
from funct_metrics import *


def train_one_epoch(model, train_loader, optimizer, criterion, device_id):
    # Put model into train mode
    model = model.train()
    print(len(train_loader))
    for train_index, (train_data, train_labels) in enumerate(train_loader):
        # --- Training begins --- #
        # Send data to gpu, if there is GPU
        # print("train_labels", train_labels.shape)
        if torch.cuda.is_available():
            train_data = train_data.cuda(device_id)
            train_labels = train_labels.cuda(device_id)
        train_labels = np.reshape(train_labels,(1000,1))
        # print("train_labels", train_labels.shape)
        # Zero gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(train_data)

        # print(outputs)
        # print(train_labels)
        # Calculate loss
        loss = criterion(outputs, train_labels.to(torch.float))
        # Backward pass
        loss.backward()
        # Update gradients
        optimizer.step()
        # --- Training ends --- #


def make_prediction(model, data_loader, device_id):
    prediction_list = []
    label_list = []

    # Put model into evaluation mode
    model = model.eval()

    for data_index, (data, labels) in enumerate(data_loader):
        # Send data to gpu
        if torch.cuda.is_available():
            data = data.cuda(device_id)
        # Forward pass without gradients
        with torch.no_grad():
            outputs = model(data)
        # If the model is in GPU, get data to cpu
        if torch.cuda.is_available():
            outputs = outputs.cpu()

        # Add predictions and labels to respective lists
        preds = torch.argmax(outputs, dim=1)
        label_list.extend(labels.tolist())
        prediction_list.extend(preds.tolist())
    return np.array(prediction_list), np.array(label_list)


if __name__ == "__main__":
    # Group-ID - don't forget to fill here
    group_id = None
    assert isinstance(1, int), 'Dont forget to add your group id'
    device_id = 1 % 2
    if torch.cuda.is_available():
        print('Using GPU:', device_id)
        print('Warning: If you are using a server with a single gpu')
        print('Manually change device_id to 0 at line 64')

    # Hyperparameters
    # -----> Tune hyperparameters here
    learning_rate = 9.99
    momentum = 9.99
    weight_decay = 9.9999
    batch_size = 1000
    num_epoch = 10
    # -----> Tune hyperparameters here

    # Define datasets and data loaders
    tr_dataset = SeqDataset(['label1_train.txt',
                             'label0_train.txt'])
    tr_loader = DataLoader(dataset=tr_dataset, batch_size=batch_size,
                           shuffle=True, num_workers=1, drop_last=True)

    ts_dataset = SeqDataset(['label1_test.txt',
                             'label0_test.txt'])
    ts_loader = DataLoader(dataset=ts_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=1, drop_last=True)

    # Model
    model = ConvolutionalNetwork()
    if torch.cuda.is_available():
        model = model.cuda(device_id)

    # Loss
    # For additional losses see: https://pytorch.org/docs/stable/nn.html
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    # For additional optimizers see: https://pytorch.org/docs/stable/optim.html
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                   weight_decay=weight_decay,
                                momentum=momentum)
    for epoch in range(num_epoch):
        print('Epoch: ', epoch, 'starts')
        # Train the model
        train_one_epoch(model, tr_loader, optimizer, criterion, device_id)
        # Make prediction
        preds, labels = make_prediction(model, ts_loader, device_id)

        # -----> Calculate metrics here
        #
        #
        # -----> Calculate metrics here

        # -----> Save the model performance here
        #
        #
        # -----> Save the model performance here
        print('Epoch: ', epoch, 'ends')

    torch.save(model.cpu(), 'my_model.pth')
    # Save the model
