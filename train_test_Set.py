import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import random

chunk=[]
with open("chr19.txt") as f:
    sequence = f.read()
    sequence=sequence.replace('\n', '').lower()
    for i in range(0, len(sequence) - (len(sequence) % 101), 101):
        chunk.append(sequence[i:i + 101])

f.close()
sequence_length= len(sequence)
print(sequence_length,len(chunk))

df = pd.read_csv("chr19.csv", header=None)
positions = df.values.tolist()

labels = np.zeros(sequence_length)
print(len(labels))

finallabel=[]
for start, end in positions:
    labels[start - 1:end] = [1] * (end - start + 1)
print(len(labels))

for i in range(0, sequence_length - (sequence_length % 101), 101):
    finallabel.append("label=1" if sum(labels[i:i+101]) > 50 else "label=0")
print(len(finallabel))

no_n_chunks = [chunk[i] for i in range(len(chunk)) if 'n' not in chunk[i]]
resultlabels = [finallabel[i] for i in range(len(chunk)) if 'n' not in chunk[i]]
print(len(no_n_chunks))
label0 = []
label1 = []

for chunk, label in zip(no_n_chunks, resultlabels):
    if label == "label=1":
        label1.append([chunk, 1])
    else:
        label0.append([chunk, 0])
print(len(label0),len(label1))
labels = ['label0', 'label1']
counts = [len(label0),len(label1)]

plt.bar(labels, counts, color=['blue', 'red'])
for i in range(len(labels)):
    plt.text(labels[i], counts[i], str(counts[i]), ha='center', va='bottom')

plt.xlabel('Labels')
plt.ylabel('Counts')
plt.title('Count of label0 and label1')
plt.show()

'''
randomlabel0 = random.sample(label0,len(label0))
randomlabel1= random.sample(label1,len(label1))


def random_split_data_to_file(data, test_ratio, prefix):
    test_size = int(test_ratio * len(data))
    vali_size = int(test_ratio * len(data))
    test_set = data[:test_size]
    vali_set = data[test_size:test_size+vali_size]
    train_set= data[test_size+vali_size:]
    print(len(test_set))
    print(len(train_set))
    print(len(vali_set))

    with open(f"{prefix}_test.txt", "w+") as file:
        for item in test_set:
            file.write(item[0] + ", " + str(item[1]) + "\n")

    with open(f"{prefix}_train.txt", "w+") as file:
        for item in train_set:
            file.write(item[0] + ", " + str(item[1]) + "\n")

    with open(f"{prefix}_validation.txt", "w+") as file:
        for item in vali_set:
            file.write(item[0] + ", " + str(item[1]) + "\n")

random_split_data_to_file(randomlabel0, 0.2, "label019default")
random_split_data_to_file(randomlabel1, 0.2, "label119default")

'''