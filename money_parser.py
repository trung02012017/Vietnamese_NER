import re


re_normalize_space = re.compile(r' +')
re_normalize_space_2 = re.compile(r'(?P<number1>[\d]) +(?P<number2>[\d])')

re_detect_money_1 = re.compile(r'\d+(\.\d+)* *(triệu|trieu|trăm|nghìn|ngìn|ngàn|mươi|cành|chai|chục|lít|củ|tỷ|tr|t|k)'
                            r' *\d* *(triệu|trieu|trăm|nghìn|ngìn|ngàn|mươi|cành|chai|chục|rưỡi|lít|củ|tỷ|tr|t|k)*'
                            r' *\d* *(triệu|trieu|trăm|nghìn|ngìn|ngàn|mươi|cành|chai|chục|rưỡi|lít|củ|tỷ|tr|t|k)*'
                            r' *\d* *(triệu|trieu|trăm|nghìn|ngìn|ngàn|mươi|cành|chai|chục|rưỡi|lít|củ|tỷ|tr|t|k)*'
                            r' *\d* *(triệu|trieu|trăm|nghìn|ngìn|ngàn|mươi|cành|chai|chục|rưỡi|lít|củ|tỷ|tr|t|k)*'
                            r' *\d* *(triệu|trieu|trăm|nghìn|ngìn|ngàn|mươi|cành|chai|chục|rưỡi|lít|củ|tỷ|tr|t|k)*'
                            r' *\d* *(triệu|trieu|trăm|nghìn|ngìn|ngàn|mươi|cành|chai|chục|rưỡi|lít|củ|tỷ|tr|t|k)*',
                               flags=re.IGNORECASE)
re_detect_money_2 = re.compile(r'\d{7,9}')
re_detect_money_3 = re.compile(r'\d{3,6} *\$')

detect_money = [re_detect_money_1, re_detect_money_2, re_detect_money_3]

map_table = {'tỷ':1e9,
             'triệu':1e6, 'tr':1e6, 't':1e6, 'trieu':1e6,
             'củ':1e6, 'cành':1e5, 'chai':1e6, 'lít':1e5,
             'k':1e3, 'nghìn':1e3, 'ngìn':1e3, 'ngàn':1e3,
             'trăm':1e2, 'mươi':10, 'chục':10, '.':0.1,
             '$': 23000}

map_table_2 = {}

stoi_map = {'một':'1', 'mốt':'1', 'hai':'2', 'ba':'3', 'bốn':'4', 'năm':'5', 'lăm':'5',
            'sáu':'6', 'bảy':'7', 'bẩy':'7', 'tám':'8', 'chín':'9'}
itos_map = {'1':'một', '2':'hai', '3':'ba', '4':'bốn', '5':'năm',
            '6':'sáu', '7':'bảy', '8':'tám', '9':'chín'}


def parse(s):
    global detect_money
    try:
        ss = s.strip().lower().replace(',', '')
        ss = stoi_ex(ss)
        ss = re_normalize_space_2.sub('\g<number1>\g<number2>', ss)
        for obj in detect_money:
            v = get_value(obj, ss)
            if v[1] is not None:
                break
        return v
    except:
        return [None, None, None]


def get_value(reobj, ss):
    try:
        value_str_raw = None
        finditer = reobj.finditer(ss)
        for m in finditer:
            x = m.regs[0]
            value_str_raw = re_normalize_space.sub(' ', ss[x[0]:x[1]])
            break

        value = stoi(value_str_raw)
        value_str_raw = value_str_raw.replace('1 mươi', 'mười')
        formatted_value = "{:,}".format(value)

        normalized_value_str = itos(formatted_value)

        return [value_str_raw, value, formatted_value, normalized_value_str]
    except:
        return [None, None, None, None]


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
        new_str = re_normalize_space.sub(' ', new_str)
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
            elif previous_unit == 1e3:
                value += number
            else:
                _ = float(words[-1])
                value += number * unit / 10
        except:
            pass
        return int(value)
    except:
        return None


def itos(s):
    try:
        result = []
        parts = s.split(',')[::-1]
        if len(parts) > 4:
            return None
        for i, p in enumerate(parts):
            if p == '000':
                continue
            s = itos_ex(p)
            if s is None:
                return None
            if i == 0:
                s += ' đồng'
            elif i == 1:
                s += ' nghìn'
            elif i == 2:
                s += ' triệu'
            elif i == 3:
                s += ' tỷ'
            result.append(s)
        return ' '.join(result[::-1])
    except:
        return None


def itos_ex(s):
    global itos_map
    result = []
    try:
        v = int(s)
        if v >= 100:
            result.append(itos_map[s[0]] + ' ' + 'trăm')

            if s[1] == '0':
                if s[2] != '0':
                    result.append('linh')
                else: pass
            elif s[1] == '1':
                result.append('mười')
            else:
                result.append(itos_map[s[1]] + ' ' + 'mươi')

            if s[2] == '0':
                pass
            elif s[1] == '0' and s[2] == '1':
                result.append('một')
            elif s[1] != '0' and s[2] == '1':
                result.append('mốt')
            else:
                result.append(itos_map[s[2]])
        elif v >= 10:
            if s[1] == 1:
                result.append('mười')
            else:
                result.append(itos_map[s[1]] + ' ' + 'mươi')

            if s[2] == '0':
                pass
            elif s[1] == '0' and s[2] == '1':
                result.append('một')
            elif s[1] != '0' and s[2] == '1':
                result.append('mốt')
            else:
                result.append(itos_map[s[2]])
        else:
            result.append(itos_map[s[2]])
        return ' '.join(result)
    except:
        return None




if __name__ == '__main__':
    # s = '1tr2'
    s = 'tôi muốn vay 321k101'
    print(parse(s))