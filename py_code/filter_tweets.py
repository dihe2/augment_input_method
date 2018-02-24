#!/usr/bin/python

file_name = '../data/twitts_geo_filt_4_copy.json'
dump_file = '../data/filtered_geo.pickle'
ind_file_name = '/home/hd89cgm/cs512/fp/data/word_ind.pickle'

import json
import pickle
import re
import subprocess as sp
from string import punctuation
#import os
#import sys
#sys.path.insert(0, '/home/hd89cgm/cs512/fp/')
#sys.path.insert(0, '/home/hd89cgm/cs512/fp/py_code')
#import goo_setup

punc_list = punctuation
punc_list.replace('<', '')
punc_list.replace('>', '')


def proc_text(text_in):
    global punc_list

    text_list = text_in.encode('utf-8').strip().replace('#', ' #').replace('@', ' @').split(' ')
    out_list = [elem for elem in text_list if(not elem.__eq__(''))]
    hash_list = []
    at_list = []

    for text_counter in range(len(out_list)):
        if out_list[text_counter][:1].__eq__('#'):
            hash_list.append(out_list[text_counter][1:])
            #out_list[text_counter] = 'HASHTAGTAG'

        elif out_list[text_counter][:1].__eq__('@'):
            #out_list[text_counter] = 'ATTAGTAG'
            at_list.append(out_list[text_counter][1:])

        #elif out_list[text_counter][:4].__eq__('http'):
            #out_list[text_counter] = 'URLURL'

        elif ord(out_list[text_counter]) > 127:
            if len(out_list[text_counter]) > 3:
                range_ind = range(0, len(out_list[text_counter]), 3)
                range_ind = range_ind[1:-1]
                temp_str = out_list[text_counter]
                for ind in reversed(range_ind):
                    temp_str = temp_str[:ind] + ' ' + temp_str[ind:]

                out_list[text_counter] = temp_str

        #else:
        #    out_list[text_counter] = out_list[text_counter].lower()

    out_str = ' '.join(out_list)
    rub = sp.Popen(['ruby', '-n', '../preprocess-twitter.rb'], stdin=sp.PIPE, stdout=sp.PIPE)
    (std_output, std_err) = rub.communicate(out_str)
    std_output = std_output.strip().replace('<', ' <').lower()

    out_list = re.findall(r"[\w']+|[{}]".format(punc_list), std_output)
    out_list = [elem for elem in out_list if (not elem.__eq__(''))]

    return out_list, hash_list, at_list


def proc_text_ext(text_in):
    global punc_list
    rub = sp.Popen(['ruby', '-n', '../preprocess-twitter.rb'], stdin = sp.PIPE, stdout = sp.PIPE)
    (std_output, std_err) = rub.communicate(text_in)
    std_output = std_output.lower()
    out_list = re.findall(r"[\w']+|[{}]".format(punc_list), std_output)

    return out_list, std_err


def map_word_to_ind(word_seq, word_dict):

    ind_seq = []
    for word in word_seq:
        if(word_dict.has_key(word)):
            ind_seq.append(word_dict[word])
        else:
            ind_seq.append(word_dict['NOTINDICT'])

    return ind_seq


twi_file = open(file_name)

with open(ind_file_name, 'r') as ind_file:
    word_to_ind, _ = pickle.load(ind_file)

twi_data_coor = []
twi_data_place = []

line = twi_file.readline()

while(line):
    if(line.__eq__("\n")):
        line = twi_file.readline()
        continue

    json_ele = json.loads(line)
    if(len(json_ele) <= 1):
        line = twi_file.readline()
        continue

    if(json_ele['coordinates']):
        text_seq, hash_seq, at_seq = proc_text(json_ele['text'])
        #text_seq_ext = proc_text_ext(json_ele['text'])
        elem = {'text':map_word_to_ind(text_seq, word_to_ind),
                'hash_tag':hash_seq,
                'at_tag':at_seq,
                'created_at':str(json_ele['created_at']),
                'coord_type':str(json_ele['coordinates']['type']),
                'lat':json_ele['coordinates']['coordinates'][1],
                'lon':json_ele['coordinates']['coordinates'][0]}

        twi_data_coor.append(elem)

    if(json_ele['place']):
        if(not (str(json_ele['place']['place_type']).__eq__('city') or
           str(json_ele['place']['place_type']).__eq__('admin') or
           str(json_ele['place']['place_type']).__eq__('country'))):

            text_seq, hash_seq, at_seq = proc_text(json_ele['text'])
            elem = {'text':map_word_to_ind(text_seq, word_to_ind),
                    'hash_tag':hash_seq,
                    'at_tag':at_seq,
                    'created_at':str(json_ele['created_at']),
                    'lat':json_ele['place']['bounding_box']['coordinates'][0][0][1],
                    'lon':json_ele['place']['bounding_box']['coordinates'][0][0][0],
                    'place_name':str(json_ele['place']['full_name'])}

            twi_data_place.append(elem)

    line = twi_file.readline()

twi_file.close()

with open(dump_file,'w') as f:
    pickle.dump([twi_data_place, twi_data_coor], f)
