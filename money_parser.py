import re


normalize_space = re.compile(r' +')
normalize_space_2 = re.compile(r'(?P<number1>[\d]) +(?P<number2>[\d])')

detect_money_1 = re.compile(r'\d+(\.\d+)* *(triệu|trieu|trăm|nghìn|ngìn|ngàn|mươi|cành|chai|chục|lít|củ|tỷ|tr|t|k)'
                            r' *\d* *(triệu|trieu|trăm|nghìn|ngìn|ngàn|mươi|cành|chai|chục|rưỡi|lít|củ|tỷ|tr|t|k)*'
                            r' *\d* *(triệu|trieu|trăm|nghìn|ngìn|ngàn|mươi|cành|chai|chục|rưỡi|lít|củ|tỷ|tr|t|k)*'
                            r' *\d* *(triệu|trieu|trăm|nghìn|ngìn|ngàn|mươi|cành|chai|chục|rưỡi|lít|củ|tỷ|tr|t|k)*'
                            r' *\d* *(triệu|trieu|trăm|nghìn|ngìn|ngàn|mươi|cành|chai|chục|rưỡi|lít|củ|tỷ|tr|t|k)*'
                            r' *\d* *(triệu|trieu|trăm|nghìn|ngìn|ngàn|mươi|cành|chai|chục|rưỡi|lít|củ|tỷ|tr|t|k)*'
                            r' *\d* *(triệu|trieu|trăm|nghìn|ngìn|ngàn|mươi|cành|chai|chục|rưỡi|lít|củ|tỷ|tr|t|k)*',
                            flags=re.IGNORECASE)
detect_money_2 = re.compile(r'\d{7,9}')
detect_money_3 = re.compile(r'\d{3,6} *\$')

detect_money = [detect_money_1, detect_money_2, detect_money_3]

map_table = {'tỷ':1e9,
             'triệu':1e6, 'tr':1e6, 't':1e6, 'trieu':1e6,
             'củ':1e6, 'cành':1e5, 'chai':1e6, 'lít':1e5,
             'k':1e3, 'nghìn':1e3, 'ngìn':1e3, 'ngàn':1e3,
             'trăm':1e2, 'mươi':10, 'chục':10, '.':0.1,
             '$': 23000}

map_table_2 = {}

stoi_map = {'một':'1', 'mốt':'1', 'hai':'2', 'ba':'3', 'bốn':'4', 'năm':'5', 'lăm':'5',
            'sáu':'6', 'bảy':'7', 'bẩy':'7', 'tám':'8', 'chín':'9'}


def parse(s):
    global detect_money
    ss = s.strip().lower().replace(',', '')
    ss = stoi_ex(ss)
    ss = normalize_space_2.sub('\g<number1>\g<number2>', ss)
    for obj in detect_money:
        v = get_value(obj, ss)
        if v[1] is not None:
            return v


def get_value(reobj, ss):
    try:
        value_str = None
        finditer = reobj.finditer(ss)
        for m in finditer:
            x = m.regs[0]
            value_str = normalize_space.sub(' ', ss[x[0]:x[1]])
            break

        value = stoi(value_str)
        value_str = value_str.replace('1 mươi', 'mười')
        return [value_str, value]
    except:
        return [None, None]


def stoi_ex(raw):
    global stoi_map
    words = raw.strip().split()
    new_words = []
    for w in words:
        try:
            if w == 'mười':
                new_words.append('1 mươi')
            else:
                ww = stoi_map[w]
                new_words.append(ww)
        except:
            new_words.append(w)
    return ' '.join(new_words)


def normalize_value(str_val):
    result = ''; previous = None; current = None
    for c in str_val:
        try:
            _ = float(c)
            current = 'num'
            if previous == 'str':
                result += ' ' + c
            else:
                result += c
            previous = current
        except:
            if c == '.':
                current = 'num'
                if previous == 'str':
                    result += ' ' + c
                else:
                    result += c
                previous = current
            else:
                current = 'str'
                if previous == 'num':
                    result += ' ' + c
                else:
                    result += c
                previous = current
    return result


def stoi(str_val):
    global map_table
    try:
        new_str = normalize_value(str_val)
        new_str = normalize_space.sub(' ', new_str)
        words = new_str.strip().split()
        unit = 0; value = 0; number = 0; previous_unit = None; temp = 0; flag = False
        for w in words:
            try:
                if number != 0 and (flag or value == 0 or value >= 1e9):
                    temp = float(w)
                else:
                    number = float(w)
            except:
                if w == 'rưỡi':
                    value += 0.5 * unit
                elif w == 'mươi' or w == 'trăm':
                    unit = map_table[w]
                    if temp != 0:
                        temp *= unit
                        number += temp
                        temp = 0
                        flag = False
                    else:
                        flag = True
                        number *= unit
                elif w == 'chục':
                    unit = map_table[w]
                    if temp != 0:
                        temp *= unit
                        number += temp
                        temp = 0
                        flag = False
                    else:
                        if number == 0:
                            number = 1
                        number *= unit
                else:
                    unit = map_table[w]
                    previous_unit = unit
                    if temp != 0:
                        number += temp
                    number *= unit
                    value += number
                    number = 0
        try:
            if previous_unit is not None and words[-1] == 'trăm':
                value += number * previous_unit / 1e3
            else:
                _ = float(words[-1])
                value += number * unit / 10
        except:
            pass
        return int(value)
    except:
        return None




if __name__ == '__main__':
    # s = '500    k 2tr'
    s = 'tôi muốn vay hai mươi tỷ 4 trăm mười lăm triệu'
    print(parse(s))