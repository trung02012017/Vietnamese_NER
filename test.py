import numpy as np
import pandas as pd

from regex import Regex

from collections import Counter
from vncorenlp import VnCoreNLP
from pyvi.ViPosTagger import postagging
from unidecode import unidecode


def create_lower_data(datapath):
    test_data = open(datapath).read().split("\n")
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


if __name__ == '__main__':
    address = pd.read_csv("/home/trungtq/Documents/NER/data/list_address/Danh sách cấp xã ___13_07_2020.csv")
    address = address.fillna("")
    print(address)

    regex = Regex()
    preprocessor = VnCoreNLP("http://127.0.0.1", port=9000)
    town_set = []
    district_set = []
    city_set = []

    count_exception = 0
    location_data = []
    for i, row in address.iterrows():
        print(i)
        town = row['Tên']
        district = row['Quận Huyện']
        city = row['Tỉnh / Thành Phố']

        try:
            town_tokenize = preprocessor.annotate(town)
            town_sen = town_tokenize['sentences'][0]
            town_w = [w['form'] for w in town_sen]

            district_tokenize = preprocessor.annotate(district)
            district_sen = district_tokenize['sentences'][0]
            district_w = [w['form'] for w in district_sen]

            city_tokenize = preprocessor.annotate(city)
            city_sen = city_tokenize['sentences'][0]
            city_w = [w['form'] for w in city_sen]

            prefix_town = ["xã", "phường", "thị_trấn"]
            prefix_district = ["huyện", "thành_phố", "thị_xã", "quận"]
            prefix_city = ["tỉnh", "thành_phố"]

            if town_w[0].lower() in prefix_town and district_w[0].lower() in prefix_district \
                    and city_w[0].lower() in prefix_city and city_w[1].lower() != "Hồ Chí Minh":
                drop_prop = np.random.rand()
                if 0 <= drop_prop <= 0.25:
                    town_w[0] = town_w[0].lower()
                    district_w[0] = district_w[0].lower()
                    city_w[0] = city_w[0].lower()
                elif 0.25 < drop_prop <= 0.5:
                    town_w = town_w[1:]
                    district_w[0] = district_w[0].lower()
                    city_w[0] = city_w[0].lower()
                elif 0.5 < drop_prop <= 0.75:
                    town_w = town_w[1:]
                    district_w = district_w[1:]
                    city_w[0] = city_w[0].lower()
                elif 0.75 < drop_prop <= 1:
                    town_w = town_w[1:]
                    district_w = district_w[1:]
                    city_w = city_w[1:]

            comma_prop = np.random.rand()

            if 0 <= comma_prop <= 1/3:
                add_str = town_w + district_w + city_w
            elif 1/3 <= comma_prop < 2/3:
                add_str = town_w + district_w + [","] + city_w
            elif 2/3 <= comma_prop <= 1:
                add_str = town_w + [","] + district_w + [","] + city_w

            pos_tag = postagging(" ".join(add_str))
            regex_tag = (add_str, [regex.run(w) for w in add_str])
            labels = []

            for i, w in enumerate(pos_tag[1]):
                if i == 0:
                    labels.append("B-LOC")
                else:
                    labels.append("I-LOC")

            print(pos_tag)
            print(regex_tag)
            print(labels)

            words = pos_tag[0]
            pos_tag = pos_tag[1]
            regex_tag = regex_tag[1]

            location_data.append("\n".join(["\t".join([words[i], pos_tag[i], regex_tag[i], labels[i]])
                                            for i, _ in enumerate(labels)]))
            location_data.append("\n")

        except IndexError:
            count_exception += 1
            continue

    print(count_exception)

    with open("location_data.txt", "w") as fp:
        fp.write("\n".join(location_data))
        fp.close()
