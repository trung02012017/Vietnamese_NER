import numpy as np
from unidecode import unidecode

test_data = open("/home/trungtq/Documents/NER/vie-ner-lstm/python3_ver/Vietnamese_NER/data/newz/normalized_data/dev_sample.txt").read().split("\n")
new_test_data = []

count = 0
for line in test_data:
    if len(line.split("\t")) == 1:
        new_test_data.append(line)
    elif len(line.split("\t")) == 4:
        split_line = line.split("\t")
        split_line[0] = split_line[0].lower()
        new_test_data.append("\t".join(split_line))
    else:
        count += 1

all_data = test_data[:(len(test_data) - 1)] + new_test_data

with open("dev_sample.txt", "w") as fp:
    fp.write("\n".join(all_data))
    fp.close()