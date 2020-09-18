import re


normalize_space = re.compile(r' +')

detect_money_1 = re.compile(r'\d+(\.\d+)* *(triệu|trieu|tr|t|k|nghìn|ngìn|ngàn|trăm|mươi) *\d* *(triệu|trieu|tr|k|nghìn|ngìn|ngàn|rưỡi|trăm)* *\d* *(triệu|trieu|tr|k|nghìn|ngìn|ngàn|rưỡi|trăm)*', flags=re.IGNORECASE)
detect_money_2 = re.compile(r'\d{7,9}')
detect_money_3 = re.compile(r'\d{3,6} *\$')

detect_money = [detect_money_1, detect_money_2, detect_money_3]

map_table = {'triệu':1e6, 'tr':1e6, 't':1e6, 'trieu':1e6,
             'k':1e3, 'nghìn':1e3, 'ngìn':1e3, 'ngàn':1e3,
             'trăm':1e2, 'mươi':10, '.':0.1,
             '$': 23000}

stoi_map = {'một':'1', 'mốt':'1', 'hai':'2', 'ba':'3', 'bốn':'4', 'năm':'5', 'sáu':'6',
               'bảy':'7', 'bẩy':'7', 'tám':'8', 'chín':'9', 'mười':'10'}


def parse(s):
    global detect_money
    ss = s.strip().lower().replace(',', '')
    ss = stoi_ex(ss)
    for obj in detect_money:
        v = get_value(obj, ss)
        if v is not None:
            return v


def get_value(reobj, ss):
    try:
        value = None
        finditer = reobj.finditer(ss)
        for m in finditer:
            x = m.regs[0]
            value = normalize_space.sub(' ', ss[x[0]:x[1]])
            break

        value = stoi(value)
        return value
    except:
        return None


def stoi_ex(raw):
    global stoi_map
    words = raw.strip().split()
    new_words = []
    for w in words:
        try:
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
        unit = 0; value = 0; number = 0
        for w in words:
            try:
                number = float(w)
            except:
                if w == 'rưỡi':
                    value += 0.5 * unit
                elif w == 'mươi':
                    unit = map_table[w]
                    number *= unit
                else:
                    unit = map_table[w]
                    value += number * unit
        try:
            _ = float(new_str[-1])
            value += number * unit / 10
        except: pass
        return int(value)
    except:
        return None



if __name__ == '__main__':
    # s = 't muốn vay hai triệu rưỡi'
    s = 't muon vay 5000$'
    print(parse(s))