# -*- encoding: utf-8 -*-

from io import open
from vncorenlp import VnCoreNLP
import os
from regex import Regex
import unicodedata


r = Regex()


def mkdir(dir):
    if (os.path.exists(dir) == False):
        os.mkdir(dir)


def push_data_to_stack(stack, file_path, file_name):
    sub_folder = os.listdir(file_path)
    for element in sub_folder:
        element = file_name + '/' + element
        stack.append(element)


def normalize_data(dataset):
    nor_dir = 'normalize_data'
    mkdir(nor_dir)
    stack = os.listdir(dataset)
    print('loading data in ' + dataset)
    while (len(stack) > 0):
        file_name = stack.pop()
        file_path = dataset + '/' + file_name
        if (os.path.isdir(file_path)):  # neu la thu muc thi day vao strong stack
            push_data_to_stack(stack, file_path, file_name)
        else:
            with open(file_path, 'r', encoding='utf-8') as fr, \
                    open(nor_dir + '/' + file_name, 'w', encoding='utf-8') as fw:
                print('processing %s' % (file_path))
                sen = []; ner = []; pos = []
                for info in fr:
                    print(info)
                    info = unicodedata.normalize('NFKC', info)
                    info = info.strip().split(u'\t')
                    print(info)
                    if len(info) == 1:
                        s, n, p = normalize_per_tag(sen, ner, pos)
                        word_info = list(map(lambda x: r.run_ex(x), sen))
                        fw.write(get_string(sen, pos, word_info, n))
                        sen = sen[:0]
                        ner = ner[:0]
                    else:
                        sen.append(info[0].replace(u' ', u'_'))
                        pos.append(info[1])

                        if len(info) == 3:
                            ner.append(info[2].split(" ")[1])
                        else:
                            ner.append(info[3])


def normalize_per_tag(sen, ner, pos):
    s = []; per = []; n = []; p = []; per_pos = []
    for i in range(len(sen)):
        if u'PER' in ner[i]:
            per.append(sen[i])
            per_pos.append(pos[i])
        else:
            if len(per) != 0:
                s.append(u'_'.join(per))
                n.append(u'B-PER')
                p.append(per_pos[0])
                per = per[:0]
                per_pos = per_pos[:0]
            s.append(sen[i])
            n.append(ner[i])
            p.append(pos[i])
    if len(per) != 0:
        s.append(u'_'.join(per))
        n.append(u'B-PER')
        p.append(per_pos[0])

    return s, n, p


def get_string(word_list, pos_list, word_info, ner_list):
    s = []
    for i in range(len(word_list)):
        try:
            ss = u'\t'.join([word_list[i], pos_list[i], word_info[i], ner_list[i]])
            s.append(ss)
        except:
            print(u'ERROR - DAMN IT !!!')
    s = u'\n'.join(s) + u'\n\n'
    return s


if __name__ == '__main__':
    normalize_data('/home/trungtq/Documents/NER/data/normalize_data')