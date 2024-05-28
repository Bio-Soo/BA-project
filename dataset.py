import numpy as np
import torch
from torch.utils.data.dataset import Dataset

class SeqDataset(Dataset):
    def __init__(self, data_loc_and_label):
        # Initialization
        self.dna_data=[]
        label = []
        [negdata,posdata] = data_loc_and_label
        negdata = open(data_loc_and_label[0])
        posdata = open(data_loc_and_label[1])
        negdatas = []
        posdatas = []
        for i in negdata.readlines():
            i = i.rstrip()
            i = i.split(", ")
            negdatas.append(i[0])
            label.append(int(i[1]))
        for i in posdata.readlines():
            i = i.rstrip()
            i = i.split(", ")
            posdatas.append(i[0])
            label.append(int(i[1]))
        data = negdatas+posdatas
        for i in range(len(data)):
            # Read data
            dna_data = data[i]
            # pdb.set_trace()
            dna_data = dna_data.replace('a', '0')
            dna_data = dna_data.replace('g', '1')
            dna_data = dna_data.replace('c', '2')
            dna_data = dna_data.replace('t', '3')
            # A [1 0 0 0]
            # G [0 1 0 0]
            # C [0 0 1 0]
            # T [0 0 0 1]
            dna_data = torch.tensor([int(digit) for digit in dna_data])
            dna_data_as_ten = torch.nn.functional.one_hot(dna_data, 4).float()
            dna_data_as_ten.unsqueeze_(dim=0)
            # print(dna_data_as_ten)
            self.dna_data.append((dna_data_as_ten, label[i]))
        print('Dataset init with', len(data), 'samples')

    def __len__(self):
        return len(self.dna_data)

    def __getitem__(self, item):
        data = self.dna_data
        # print(data[item][0].shape)
        # print(data[item][1])
        return data[item]


if __name__ == '__main__':
    tis_pos = SeqDataset(["label0_test.txt",
                          "label1_test.txt"])

    x,y = tis_pos[0]
    print(x.shape)
    print(y)
    print(len(tis_pos))