import numpy as np
from unidecode import unidecode

test_data = open("/home/trungtq/Documents/NER/data/new/normalize_data/test_sample.txt").read().split("\n")
new_test_data = []

for line in test_data:
    if len(line.split("\t")) == 1:
        new_test_data.append(line)
    else:
        split_line = line.split("\t")
        split_line[0] = split_line[0].lower()
        new_test_data.append("\t".join(split_line))

print(len(test_data))
print(len(new_test_data))

for line in new_test_data:
    print(line)