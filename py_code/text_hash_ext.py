#!/usr/bin/python

twi_file_name = '/home/hd89cgm/cs512/fp/data/twitts_geo_filt_2_copy.json'
dict_dump_name = '/home/hd89cgm/cs512/fp/data/dict_dump2_ext.pickle'
word_dict_name = '/home/hd89cgm/cs512/fp/data/word_dict2_ext.pickle'
word_ind_name = '/home/hd89cgm/cs512/fp/data/word_ind2_ext_40k.pickle'
DICT_SIZE = 40000

import json
import pickle
import re
import numpy as np
from string import punctuation

punc_list = punctuation
punc_list = punc_list.replace('<', '')
punc_list = punc_list.replace('>', '')

def proc_text(text_in):

    global punc_list

    text_list = text_in.encode('utf-8').strip().replace('#', ' #').replace('@', ' @').replace('http', ' http').split(' ')
    out_list = [elem for elem in text_list if(not elem.__eq__(''))]
    #hash_list = []
    #at_list = []

    for text_counter in range(len(out_list)):
        if out_list[text_counter][:1].__eq__('#'):
            #hash_list.append(out_list[text_counter][1:])
            out_list[text_counter] = '<hashtag> ' + out_list[text_counter][1:]

        elif out_list[text_counter][:1].__eq__('@'):
            #at_list.append(out_list[text_counter][1:])
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
print 'Word count ratio outside of the dictionary: {}'.format(sum(word_count[sorted_ind[DICT_SIZE:]])/float(sum(word_count)))
print 'Max word count outside dict: {}'.format(word_count[sorted_ind[DICT_SIZE]])
print 'Max count ratio outside dict: {}'.format(word_count[sorted_ind[DICT_SIZE]]/float(sum(word_count)))

with open(word_dict_name,'w') as dict_name:
    pickle.dump(word_dict,dict_name)

word_key = word_dict.keys()
word_key.append('NOTINDICT')
word_to_ind = dict(zip(word_key, range(len(word_key))))
ind_to_word = dict(zip(range(len(word_key)), word_key))

with open(word_ind_name,'w') as ind_name:
    pickle.dump([word_to_ind, ind_to_word], ind_name)


