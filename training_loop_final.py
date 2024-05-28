import csv
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from dataset import SeqDataset  #one hot encoding
from Final_model import ConvolutionalNetwork #cnn
from sklearn.metrics import roc_curve, auc
import time


def train_one_epoch(model, train_loader,validation_loader, optimizer, criterion, device_id):
    train_true=[]
    train_preds=[]
    train_preds_prob=[]
    loss_per_epoch = 0.0
    model = model.train()
    for train_index, (train_data, train_labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            train_data = train_data.cuda(device_id)
            train_labels = train_labels.type(torch.LongTensor)
            train_labels = train_labels.cuda(device_id)
            train_labels = torch.reshape(train_labels,(batch_size,1))
            train_labels = train_labels.squeeze(dim=-1)
        optimizer.zero_grad()
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        loss_per_epoch += loss.item()
        train_true.extend(train_labels.tolist())
        preds_prob_tr = torch.sigmoid(outputs[:, 1])
        train_preds_prob.extend(preds_prob_tr.tolist())
        preds_tr = torch.round(preds_prob_tr).int()
        train_preds.extend(preds_tr.tolist())
    loss_per_epoch  /= (train_index + 1)

    acc_per_epoch = 0
    for i in range(len(train_true)):
        if train_true[i] == train_preds[i]:
            acc_per_epoch += 1
    acc_per_epoch /= (batch_size*(train_index + 1))

    val_true = []
    val_preds = []
    val_preds_prob = []
    loss_per_epoch_v = 0.0
    model = model.eval()
    for validation_index, (validation_data, validation_labels) in enumerate(validation_loader):
        if torch.cuda.is_available():
            validation_data = validation_data.cuda(device_id)
            validation_labels = validation_labels.cuda(device_id)
            validation_labels = torch.reshape(validation_labels,(batch_size,1))
            validation_labels = validation_labels.squeeze(dim=-1)
            with torch.no_grad():
                outputs = model(validation_data)
            loss = criterion(outputs, validation_labels)
            loss_per_epoch_v += loss.item()
            val_true.extend(validation_labels.tolist())
            preds_prob_val = torch.sigmoid(outputs[:, 1])
            val_preds_prob.extend(preds_prob_val.tolist())
            preds_val = torch.round(preds_prob_val).int()
            val_preds.extend(preds_val.tolist())
    loss_per_epoch_v /= (validation_index + 1)

    acc_per_epoch_v = 0
    for j in range(len(val_true)):
        if val_true[j]==val_preds[j]:
            acc_per_epoch_v+=1
    acc_per_epoch_v /= (batch_size * (validation_index + 1))

    return loss_per_epoch, acc_per_epoch, loss_per_epoch_v, acc_per_epoch_v, train_true,train_preds,train_preds_prob,val_true, val_preds, val_preds_prob

def make_prediction(model, data_loader, device_id):
    test_true = []
    test_preds = []
    test_preds_prob = []
    acc_test = 0
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
        test_true.extend(labels.tolist())
        preds_prob_ts = torch.sigmoid(outputs[:, 1])
        test_preds_prob.extend(preds_prob_ts.tolist())
        preds_ts = torch.round(preds_prob_ts).int()
        test_preds.extend(preds_ts.tolist())

    for k in range(len(test_true)):
        if test_true[k]==test_preds[k]:
            acc_test += 1
    acc_test /= ((data_index+1)*batch_size)

    return acc_test, test_true, test_preds, test_preds_prob

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
    learning_rate = 0.005   # defalut 0.01, recommended range 0.1~0.0001
    momentum = 0.0  # default 0.0
    weight_decay = 0.0
    batch_size = 1024
    num_epoch = 50

    start_time = time.time()
    # Define datasets and data loaders
    tr_dataset = SeqDataset(['label019default_train.txt',
                             'label119default_train.txt'])
    tr_loader = DataLoader(dataset=tr_dataset, batch_size=batch_size,
                           shuffle=True, num_workers=1, drop_last=True)
    tv_dataset = SeqDataset(['label019default_validation.txt',
                             'label119default_validation.txt'])
    tv_loader = DataLoader(dataset=tv_dataset, batch_size=batch_size,
                           shuffle=True, num_workers=1, drop_last=True)
    ts_dataset = SeqDataset(['label019default_test.txt',
                             'label119default_test.txt'])
    ts_loader = DataLoader(dataset=ts_dataset, batch_size=batch_size,
                           shuffle=True, num_workers=1, drop_last=True)

    model = ConvolutionalNetwork()
    if torch.cuda.is_available():
        model = model.cuda(device_id)

    # Loss
    class_weights = torch.tensor([1, 40], dtype=torch.float)
    class_weights = class_weights.cuda(device_id)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay,
                                momentum=momentum)
    loss_list_train = []
    loss_list_validation = []
    train_acc = []
    validation_acc = []
    roc_auc_train =[]
    roc_auc_validation =[]
    for epoch in range(num_epoch):
        print('Epoch: ', epoch, 'starts')
        loss_train, accuracy_train,loss_val, accuracy_val, train_true_labels, train_pred_labels, tr_pred_prob_label, val_true_labels, val_pred_labels, val_pred_prob_label, = train_one_epoch(model, tr_loader,tv_loader, optimizer, criterion, device_id)

        fprt, tprt, thresholdt = roc_curve(train_true_labels, tr_pred_prob_label)
        roc_auc_train.append(auc(fprt, tprt))
        fprv, tprv, thresholdv = roc_curve(val_true_labels, val_pred_prob_label)
        roc_auc_validation.append(auc(fprv, tprv))

        print("Train loss of {}/{} epoch : {:.7f}, Accuracy = {}".format(epoch + 1, num_epoch, loss_train, accuracy_train))
        loss_list_train.append(loss_train)
        train_acc.append(accuracy_train)
        print("Validation loss of {}/{} epoch : {:.7f}, Accuracy = {}".format(epoch + 1, num_epoch, loss_val, accuracy_val))
        loss_list_validation.append(loss_val)
        validation_acc.append(accuracy_val)
        print('Epoch: ', epoch, 'ends')

    print("Test starts")
    print("test ends")
    accuracy_test, test_true_labels, test_pred_labels, test_pred_prob_label = make_prediction(model,ts_loader,device_id)
    print("Test accuracy: {:.4f} ".format(accuracy_test))
    end_time = time.time()
    print("Training time", end_time-start_time)
    '''
    plot_losses(loss_list_train, loss_list_validation)
    plot_accuracy(train_acc, validation_acc)
    conf_matrix(test_true_labels, test_pred_labels)
    roc_curve_plot(test_true_labels, test_pred_labels)
    pr_curve_plot(test_true_labels, test_pred_labels)

    torch.save(model.cpu(), '../my_model.pth')
    '''

    with open('loss.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['train', 'validation'])
        for row in zip(loss_list_train, loss_list_validation):
            writer.writerow(row)

    with open('accuracy.csv', 'w', newline='') as csvfile1:
        writer = csv.writer(csvfile1)
        writer.writerow(['train', 'validation'])
        for row in zip(train_acc,validation_acc):
            writer.writerow(row)
    print(test_true_labels[:10],test_pred_labels[:10])

    with open('labels and prediction_prob.csv', 'w', newline='') as csvfile2:
        writer = csv.writer(csvfile2)
        writer.writerow([' test_true_labels', 'test_pred_prob'])
        for row in zip(test_true_labels,test_pred_prob_label):
            writer.writerow(row)

    with open('labels and prediction.csv', 'w', newline='') as csvfile3:
        writer = csv.writer(csvfile3)
        writer.writerow([' test_true_labels', 'test_pred_labels'])
        for row in zip(test_true_labels,test_pred_labels):
            writer.writerow(row)

    with open('ROC_AUC_per_epoch.csv', 'w', newline='') as csvfile4:
        writer = csv.writer(csvfile4)
        writer.writerow(['train', 'validation'])
        for row in zip(roc_auc_train, roc_auc_validation):
            writer.writerow(row)