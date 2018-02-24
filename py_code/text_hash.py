#!/usr/bin/python

twi_file_name = '/home/hd89cgm/cs512/fp/data/twitts_geo_filt_4_copy.json'
dict_dump_name = '/home/hd89cgm/cs512/fp/data/dict_dump.pickle'
word_dict_name = '/home/hd89cgm/cs512/fp/data/word_dict.pickle'
word_ind_name = '/home/hd89cgm/cs512/fp/data/word_ind.pickle'
DICT_SIZE = 1000

import json
import pickle
import re
import numpy as np

def proc_text(text_in):
    text_list = text_in.encode('utf-8').strip().replace('\xf0', ' ').replace('\xe2', ' ').split(' ')
    out_list = [elem for elem in text_list if(not elem.__eq__(''))]
    for text_counter in range(len(out_list)):
        if out_list[text_counter][:1].__eq__('#'):
            out_list[text_counter] = 'HASHTAGTAG'

        elif out_list[text_counter][:1].__eq__('@'):
            out_list[text_counter] = 'ATTAGTAG'

        elif out_list[text_counter][:4].__eq__('http'):
            out_list[text_counter] = 'URLURL'

        else:
            out_list[text_counter] = out_list[text_counter].lower()

    out_str = ' '.join(out_list)
    out_list = re.findall(r"[\w']+|[.,!?;]", out_str)

    return out_list


twi_file = open(twi_file_name, 'r')

word_table = {}
line = twi_file.readline()

while(line):
    if(line.__eq__("\n")):
        line = twi_file.readline()
        continue

    json_ele = json.loads(line)
    if(len(json_ele) <= 1):
        line = twi_file.readline()
        continue

    if(json_ele['lang']):
        if(not str(json_ele['lang']).__eq__('en')):
            line = twi_file.readline()
            continue
    else:
        line = twi_file.readline()
        continue

    if(json_ele['text']):
        word_seq = proc_text(json_ele['text'])
        for word in word_seq:
            if word_table.has_key(word):
                word_table[word] += 1
            else:
                word_table[word] = 1

    line = twi_file.readline()


twi_file.close()

#with open(dict_dump_name,'w') as dump_file:
#    pickle.dump(word_table,dump_file)

word_count = np.array(word_table.values())
word_key = np.array(word_table.keys())
sorted_ind = np.argsort(-1*word_count)

#word_keys = np.array(word_table.keys())
#word_count = np.array(word_table.values())

word_dict = dict(zip(word_key[sorted_ind[:DICT_SIZE]], word_count[sorted_ind[:DICT_SIZE]]))

with open(word_dict_name,'w') as dict_name:
    pickle.dump(word_dict,dict_name)

word_key = word_dict.keys()
word_key.append('NOTINDICT')
word_to_ind = dict(zip(word_key, range(len(word_key))))
ind_to_word = dict(zip(range(len(word_key)), word_key))

with open(word_ind_name,'w') as ind_name:
    pickle.dump([word_to_ind, ind_to_word], ind_name)


