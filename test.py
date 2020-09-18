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


def create_location_data():
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


if __name__ == '__main__':
    from vncorenlp import VnCoreNLP

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
            pos_tags = [w['posTag'] for w in sen[0]]
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
