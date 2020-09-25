import re
import json

from unidecode import unidecode
from datetime import datetime


def create_motor_data():
    motor_data = open("/home/trungtq/Documents/NER/data/motor_data.txt").read().split("\n\n")

    motor_data_dict = {}

    for line in motor_data:
        word_lines = line.split("\n")
        brand_line = word_lines[0].split("\t")
        year_line = word_lines[-1].split("\t")

        if brand_line[-1] == 'BRAND':
            brand_name = brand_line[0].lower()
            try:
                model_dict = motor_data_dict[brand_name]
            except KeyError:
                motor_data_dict[brand_name] = {}

            if 3 <= len(word_lines) <= 4 or len(word_lines) == 6:
                b_model_line = word_lines[1]
                b_model_name = b_model_line.split("\t")[0].lower()
                try:
                    i_model_dict = motor_data_dict[brand_name][b_model_name]
                except KeyError:
                    motor_data_dict[brand_name][b_model_name] = set()

                if len(word_lines) == 4:
                    i_model_line = word_lines[2].split("\t")
                    i_model_name = i_model_line[0].lower()
                    motor_data_dict[brand_name][b_model_name].add(i_model_name)
                else:
                    pass
            else:
                print(word_lines)
                print("Format fail")
        else:
            print("No brand")

    for k, v in motor_data_dict.items():
        for k_v, v_v in v.items():
            motor_data_dict[k][k_v] = list(motor_data_dict[k][k_v])

    with open("motor_data.json", "w") as fp:
        json.dump(motor_data_dict, fp)
        fp.close()


def choose_extra(choose_list, str_):
    if len(choose_list) == 0:
        return ""
    idx_in_string = [str_.index(w) for w in choose_list]
    smallest_idx_in_str = min(idx_in_string)
    index_smallest = [element for idx, element in enumerate(idx_in_string) if smallest_idx_in_str == idx_in_string[idx]]
    if len(index_smallest) == 1:
        return choose_list[idx_in_string.index(index_smallest[0])]
    else:
        choose_list = [choose_list[idx] for idx in index_smallest]
        return max(choose_list, key=len)


def detect_brand_and_model(str_, data):
    motor_brand_data = data[0]
    motor_b_model_data = data[1]
    motor_i_model_data = data[2]
    motor_brand, b_model, i_model = "", "", ""

    motor_brands = "|".join(list(motor_brand_data.keys()))
    detect_motor_brand = re.findall(r'(' + motor_brands + ')', str_, flags=re.IGNORECASE)

    motor_b_models = "|".join(list(motor_b_model_data.keys()))
    detect_motor_b_model = re.findall(r'(' + motor_b_models + ')', str_, flags=re.IGNORECASE)

    detect_motor_i_model = []
    for i_model_ in list(motor_i_model_data.keys()):
        if len(re.findall(r'(' + i_model_ + ')', str_, flags=re.IGNORECASE)):
            detect_motor_i_model.append(i_model_)

    if len(detect_motor_brand) > 0:  # got brand
        motor_brand = detect_motor_brand[0]
        b_model_list = motor_brand_data[motor_brand].keys()

        satisfied_b_models = [m for m in detect_motor_b_model if m in b_model_list]
        if len(satisfied_b_models) > 0:  # got brand and b_model
            b_model = choose_extra(satisfied_b_models, str_)
            i_model_list = motor_brand_data[motor_brand][b_model]
            satisfied_i_models = [m for m in i_model_list if m in detect_motor_i_model]
            if len(satisfied_i_models) > 0:  # got brand, b_model and i_model
                i_model = choose_extra(satisfied_i_models, str_)
            else:  # got brand, b_model but no i_model
                pass
        else:  # got brand and no b_model, i_model considered
            i_model_list = []
            for k, v in motor_brand_data[motor_brand].items():
                i_model_list += v
            satisfied_i_models = [m for m in i_model_list if m in detect_motor_i_model]
            if len(satisfied_i_models) > 0:
                i_model = choose_extra(satisfied_i_models, str_)
                b_model = motor_i_model_data[i_model][b_model]
            else:
                pass
    else:  # no brand
        b_model_list = list(motor_b_model_data.keys())
        satisfied_b_models = [m for m in detect_motor_b_model if m in b_model_list]

        if len(satisfied_b_models) > 0:  # no brand and got b_model
            b_model = choose_extra(satisfied_b_models, str_)
            motor_brand = motor_b_model_data[b_model]['brand']
            i_model_list = motor_b_model_data[b_model]['i_model']

            satisfied_i_models = [m for m in i_model_list if m in detect_motor_i_model]
            if len(satisfied_i_models) > 0:  # no brand, got b_model and i_model
                i_model = choose_extra(satisfied_i_models, str_)
            else:  # got brand, b_model but no i_model
                pass

        else:  # no brand, no b_model, i_model considered
            i_model_list = []
            for brand_, v_brand in motor_brand_data.items():
                for b_model_, i_model_ in v_brand.items():
                    i_model_list += i_model_
            satisfied_i_models = [m for m in i_model_list if m in detect_motor_i_model]
            if len(satisfied_i_models) > 0:  # no brand, no b_model and got i_model
                i_model = choose_extra(satisfied_i_models, str_)
                motor_brand = motor_i_model_data[i_model]['brand']
                b_model = motor_i_model_data[i_model]['b_model']
            else:  # no brand, no b_model and no i_model
                pass
    return motor_brand, b_model, i_model


def detect_year_released(str_):
    year_released = 0
    detect_year_released = re.findall(r'[1-2][0-9]{3}', str_)

    for y in detect_year_released:
        year_released = int(y)
        if 1900 <= year_released <= datetime.now().year:
            break
        else:
            pass
    return year_released


def detect_i_model_bonus(str_, i_model, year_released):
    i_model_bonus = []
    detect_i_model_bonus = re.findall(r'[0-9]{3}', str_)
    for z in detect_i_model_bonus:
        if 90 <= int(z) <= 300 and z not in i_model and z not in str(year_released):
            i_model_bonus.append(z)
            break
        else:
            pass

    if "nhap khau" in unidecode(str_.lower()):
        i_model_bonus.append("nhập khẩu")
    return i_model_bonus


def create_data_mapping(brand_mapping_path='motor_data.json'):
    motor_brand_data = json.load(open(brand_mapping_path, "r"))
    motor_b_model_data = {}
    motor_i_model_data = {}

    for k, v in motor_brand_data.items():
        for b_model in v.keys():
            motor_b_model_data[b_model] = {
                "brand": k,
                "i_model": v[b_model]
            }

            for i_model in v[b_model]:
                motor_i_model_data[i_model] = {
                    "brand": k,
                    "b_model": b_model
                }

    return motor_brand_data, motor_b_model_data, motor_i_model_data


def main_process_motor(str_):
    mapping_data = create_data_mapping()
    motor_brand, motor_b_model, motor_i_model = detect_brand_and_model(str_, mapping_data)
    year_released = detect_year_released(str_)
    motor_i_model_bonus = detect_i_model_bonus(str_, motor_i_model, year_released)

    result = {
        "motor_brand": motor_brand,
        "motor_b_model": motor_b_model,
        "motor_i_model": motor_i_model,
        "year_released": year_released,
        "motor_i_model_bonus": motor_i_model_bonus
    }

    return result


print(main_process_motor("like fi 2020"))

