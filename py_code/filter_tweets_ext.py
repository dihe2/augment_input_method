#!/usr/bin/python

#file_name = '../data/twitts_geo_filt_2_copy.json'
file_name = '../data/twitts_geo_filt.json'
#dump_file = '../data/filtered_geo_ext_2.pickle'
dump_file = '../data/filtered_geo_ext_full.pickle'
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
punc_list = punc_list.replace('<', '')
punc_list = punc_list.replace('>', '')

def proc_text(text_in):

    global punc_list

    text_list = text_in.encode('utf-8').strip().replace('#', ' #').replace('@', ' @').replace('http', ' http').split(' ')
    out_list = [elem for elem in text_list if(not elem.__eq__(''))]
    hash_list = []
    at_list = []

    for text_counter in range(len(out_list)):
        if out_list[text_counter][:1].__eq__('#'):
            hash_list.append(out_list[text_counter][1:])
            out_list[text_counter] = '<hashtag> ' + out_list[text_counter][1:]

        elif out_list[text_counter][:1].__eq__('@'):
            at_list.append(out_list[text_counter][1:])
            out_list[text_counter] = '<user> ' + out_list[text_counter][1:]

        elif out_list[text_counter][:4].__eq__('http'):
            out_list[text_counter] = '<url>'

        elif len(out_list[text_counter]) > 2 and re.match('[a-zA-Z]]', out_list[text_counter][-1]):
            if (out_list[text_counter][-1] == out_list[text_counter][-2] and
                        out_list[text_counter][-2] == out_list[text_counter][-3]):

                break_ind = -1
                for elong_ind in reversed(range(len(out_list[text_counter]) - 1)):
                    if not out_list[text_counter][elong_ind] == out_list[text_counter][-1]:
                        break_ind = elong_ind
                        break

                out_list[text_counter] = out_list[text_counter][:break_ind + 2] + ' <elong>'

        elif len(out_list[text_counter]) > 1 and re.match('[.!?]', out_list[text_counter][-1]):
            if out_list[text_counter][-1].__eq__(out_list[text_counter][-2]):

                break_ind = - 1
                for elong_ind in reversed(range(len(out_list[text_counter]) - 1)):
                    if not out_list[text_counter][elong_ind] == out_list[text_counter][-1]:
                        break_ind = elong_ind
                        break

                out_list[text_counter] = out_list[text_counter][:break_ind + 2] + ' <repeat>'

        elif len(re.split("([0-9]+)", out_list[text_counter])) > 1 or re.match("[0-9]+", out_list[text_counter]):
            num_split = re.split("([0-9]+)", out_list[text_counter])
            num_split = [ele for ele in num_split if not ele.__eq__('')]
            for ele_ind in range(len(num_split)):
                if re.match("[0-9]+", num_split[ele_ind]):
                    num_split[ele_ind] = '<NUMBER>'

            #if len(num_split) > 1:
            out_list[text_counter] = ' '.join(num_split)
            #else:
            #    out_list[text_counter] = num_split

    #print out_list

    for text_counter in range(len(out_list)):
        temp_word = out_list[text_counter]

        char_counter = 0
        emoj_list = []
        last_emoj = 0
        while char_counter < len(temp_word):
            if ord(temp_word[char_counter]) > 127:
                if char_counter > 0:
                    emoj_list.append(temp_word[last_emoj:char_counter])
                    emoj_list.append(temp_word[char_counter:char_counter + 3])
                    char_counter += 3
                    last_emoj = char_counter
                else:
                    emoj_list.append(temp_word[char_counter:char_counter + 3])
                    char_counter += 3
                    last_emoj = char_counter

            else:
                char_counter += 1

        if len(emoj_list) == 0:
            out_list[text_counter] = temp_word
        else:
            if last_emoj < len(temp_word):
                emoj_list.append(temp_word[last_emoj:])

            out_list[text_counter] = ' '.join(emoj_list)

        #else:
        #    out_list[text_counter] = out_list[text_counter].lower()

    out_str = ' '.join(out_list).lower()
    #rub = sp.Popen(['ruby', '-n', '../preprocess-twitter.rb'], stdin=sp.PIPE, stdout=sp.PIPE)
    #(std_output, std_err) = rub.communicate(out_str)
    #std_output = std_output.strip().replace('<', ' <').lower()

    out_list = re.findall(r"[\w']+|<[\w']+>|[{}]".format(punc_list), out_str)
    out_list = [elem for elem in out_list if (not elem.__eq__(''))]

    #print out_list
    #print punc_list

    return out_list, hash_list, at_list


#def proc_text_ext(text_in):
#    global punc_list
#    rub = sp.Popen(['ruby', '-n', '../preprocess-twitter.rb'], stdin = sp.PIPE, stdout = sp.PIPE)
#    (std_output, std_err) = rub.communicate(text_in)
#    std_output = std_output.lower()
#    out_list = re.findall(r"[\w']+|[{}]".format(punc_list), std_output)
#
#    return out_list, std_err


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

error_lines = []
line_counter = 0

line = twi_file.readline()

while(line):
    line_counter += 1
    if(line.__eq__("\n")):
        line = twi_file.readline()
        continue

    try:
        json_ele = json.loads(line)
    except ValueError:
        print 'error on line {}'.format(line_counter)
        error_lines.append(line_counter)
        line = twi_file.readline()
        continue

    if(len(json_ele) <= 1):
        line = twi_file.readline()
        continue

    if(json_ele['coordinates']):
        text_seq_ext, hash_seq, at_seq = proc_text(json_ele['text'])
        #text_seq_ext, err = proc_text_ext(json_ele['text'])
        #if std_err:
        #    print 'ERROR'
        #    print json_ele['text']

        elem = {'text':text_seq_ext,
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

            text_seq_ext, hash_seq, at_seq = proc_text(json_ele['text'])
            #text_seq_ext, err = proc_text_ext(json_ele['text'])
            #if std_err:
            #    print 'ERROR'
            #    print json_ele['text']

            elem = {'text':text_seq_ext,
                    'hash_tag':hash_seq,
                    'at_tag':at_seq,
                    'created_at':str(json_ele['created_at']),
                    'lat':json_ele['place']['bounding_box']['coordinates'][0][0][1],
                    'lon':json_ele['place']['bounding_box']['coordinates'][0][0][0],
                    'place_name':json_ele['place']['full_name'].encode('utf-8')}

            twi_data_place.append(elem)

    line = twi_file.readline()

twi_file.close()

with open(dump_file,'w') as f:
    pickle.dump([twi_data_place, twi_data_coor], f)
