import psycopg2

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


def create_location_sentence(is_prefix, is_comma,
                             town, district, city,
                             prefix_town, prefix_district, prefix_city):
    if is_prefix:
        pass
    else:
        if any([w in town for w in prefix_town]):
            for w in prefix_town:
                town = town.replace(w, "").strip()

        if any([w in district for w in prefix_district]):
            for w in prefix_district:
                district = district.replace(w, "").strip()

        if any([w in city for w in prefix_city]):
            for w in prefix_city:
                city = city.replace(w, "").strip()

    if is_comma:
        add_str = ", ".join([town, district, city])
    else:
        add_str = " ".join([town, district, city])

    return add_str


def create_location_data():
    address = pd.read_csv("/home/trungtq/Documents/NER/data/list_address/Danh sách cấp xã ___13_07_2020.csv")
    address = address.fillna("")
    print(address)

    regex = Regex()
    preprocessor = VnCoreNLP("http://127.0.0.1", port=9000)

    location_data = []
    for i, row in address.iterrows():
        print(i)
        if len(row['Tên'].split()) > 0:
            town = " ".join([row['Tên'].split()[0].lower()] + row['Tên'].split()[1:])
        else:
            town = row['Tên']

        if not row['Quận Huyện'].split()[1].isdigit():
            district = " ".join([row['Quận Huyện'].split()[0].lower()] + row['Quận Huyện'].split()[1:])
        else:
            district = row['Quận Huyện']
        city = " ".join([row['Tỉnh / Thành Phố'].split()[0].lower()] + row['Tỉnh / Thành Phố'].split()[1:])

        prefix_town = ["xã", "phường", "thị trấn"]
        prefix_district = ["huyện", "thành phố", "thị xã", "quận"]
        prefix_city = ["tỉnh", "thành phố"]

        params = [town, district, city, prefix_town, prefix_district, prefix_city]
        # add_str_1 = create_location_sentence(True, True, *params)
        # add_str_2 = create_location_sentence(True, False, *params)
        add_str_3 = create_location_sentence(False, True, *params)
        add_str_4 = create_location_sentence(False, False, *params)

        # for add_str in [add_str_1, add_str_2, add_str_3, add_str_4]:
        for add_str in [add_str_3, add_str_4]:
            pre_result = preprocessor.annotate(add_str)['sentences'][0]
            words = [w['form'] for w in pre_result]
            pos_tags = postagging(" ".join(words))[1]
            regexes = [regex.run(w) for w in words]
            labels = []
            for idx_w, w in enumerate(words):
                if idx_w == 0:
                    labels.append("B-LOC")
                else:
                    labels.append("I-LOC")

            for idx_w, w in enumerate(words):
                line = "\t".join([w, pos_tags[idx_w], regexes[idx_w], labels[idx_w]])
                location_data.append(line)

            location_data.append("")

    with open("location_data.txt", "w") as fp:
        fp.write("\n".join(location_data))
        fp.close()


def connect_db_postgre_moto_info():

    db_host = "172.16.30.240"
    port = 5432
    db_name = "credit_score"
    user = "credit_score"
    password = "lbHoKPMYyuc5LO4z"

    conn = psycopg2.connect(host=db_host, port=port, database=db_name, user=user, password=password)
    return conn


def query_postgre_motors(conn, sql_query):
    return pd.read_sql(sql_query, conn)


def create_motor_data():
    regex = Regex()
    annotator = VnCoreNLP(address="http://127.0.0.1", port=9000)
    conn_moto_info = connect_db_postgre_moto_info()

    query = """select * from motors"""
    df = query_postgre_motors(conn_moto_info, query)

    print(df)

    moto_info_data = []
    for i, row in df.iterrows():
        first_year = row['first_year']
        last_year = row['last_year']
        year = first_year
        while year < last_year + 1:
            text = " ".join([row['brand'], row['model'], str(year)])
            sen = annotator.annotate(text)['sentences']
            words_1 = [w['form'] for w in sen[0]]
            words_2 = [w['form'].upper() for w in sen[0]]
            words_3 = [w['form'].title() for w in sen[0]]
            pos_tags = postagging(" ".join([w['form'] for w in sen[0]]))[1]
            regexes = [regex.run(w) for w in words_1]

            for w_idx, word in enumerate(words_1):
                if w_idx == 0:
                    line = word + "\t" + pos_tags[w_idx] + "\t" + regexes[w_idx] + "\t" + "BRAND"
                elif w_idx == len(words_1) - 1:
                    line = word + "\t" + pos_tags[w_idx] + "\t" + regexes[w_idx] + "\t" + "YEAR"
                elif w_idx == 1:
                    line = word + "\t" + pos_tags[w_idx] + "\t" + regexes[w_idx] + "\t" + "B-MODEL"
                else:
                    line = word + "\t" + pos_tags[w_idx] + "\t" + regexes[w_idx] + "\t" + "I-MODEL"
                moto_info_data.append(line)
            moto_info_data.append("")

            for w_idx, word in enumerate(words_2):
                if w_idx == 0:
                    line = word + "\t" + pos_tags[w_idx] + "\t" + regexes[w_idx] + "\t" + "BRAND"
                elif w_idx == len(words_2) - 1:
                    line = word + "\t" + pos_tags[w_idx] + "\t" + regexes[w_idx] + "\t" + "YEAR"
                elif w_idx == 1:
                    line = word + "\t" + pos_tags[w_idx] + "\t" + regexes[w_idx] + "\t" + "B-MODEL"
                else:
                    line = word + "\t" + pos_tags[w_idx] + "\t" + regexes[w_idx] + "\t" + "I-MODEL"
                moto_info_data.append(line)
            moto_info_data.append("")

            for w_idx, word in enumerate(words_3):
                if w_idx == 0:
                    line = word + "\t" + pos_tags[w_idx] + "\t" + regexes[w_idx] + "\t" + "BRAND"
                elif w_idx == len(words_3) - 1:
                    line = word + "\t" + pos_tags[w_idx] + "\t" + regexes[w_idx] + "\t" + "YEAR"
                elif w_idx == 1:
                    line = word + "\t" + pos_tags[w_idx] + "\t" + regexes[w_idx] + "\t" + "B-MODEL"
                else:
                    line = word + "\t" + pos_tags[w_idx] + "\t" + regexes[w_idx] + "\t" + "I-MODEL"
                moto_info_data.append(line)
            moto_info_data.append("")
            year += 1

    with open("motor_data.txt", "w") as fp:
        text = "\n".join(moto_info_data)
        fp.write(text)
        fp.close()


if __name__ == '__main__':
    from random import shuffle

    root_data_train = open("/home/trungtq/Documents/NER/data/root/train_sample.txt").read().split("\n\n")
    location_data_train = open("/home/trungtq/Documents/NER/data/location_data.txt").read().split("\n\n")
    motor_data_train = open("/home/trungtq/Documents/NER/data/motor_data.txt").read().split("\n\n")

    data_train = root_data_train + location_data_train

    with open("train_sample.txt", "w") as fp:
        fp.write("\n\n".join(data_train))
        fp.close()