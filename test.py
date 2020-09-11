import numpy as np
from unidecode import unidecode

test_data = open("/home/trungtq/Documents/NER/data/new/normalize_data/val_sample.txt").read().split("\n")
new_test_data = []

for line in test_data:
    if len(line.split("\t")) == 1:
        new_test_data.append(line)
    else:
        split_line = line.split("\t")
        if np.random.rand() < 0.4:
            split_line[0] = unidecode(split_line[0])
        new_test_data.append("\t".join(split_line))

test_data = test_data[0:len(test_data) - 1]
all_test_data = test_data + new_test_data

print(len(all_test_data))

with open("val_data.txt", "w") as fp:
    fp.write("\n".join(all_test_data))
    fp.close()