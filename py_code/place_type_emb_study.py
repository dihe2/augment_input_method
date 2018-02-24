#!/usr/bin/python

MIN_SUP = 5
TOP_FREQ_WORD = 50
PRINT_WORD_COUNT = 10
SAMPLE_COUNT = 200
SAMPLE_TOTAL_COUNT = 500
EMB_DIM = 100

emb_dict_file = '/home/hd89cgm/cs512/fp/data/glove.twitter.27B.100d.txt'
twe_goo_place_file = '/home/hd89cgm/cs512/fp/data/geo_place_full.pickle'
dist_phrase_file = '../results/place_dist_phrases2.txt'
place_table_file = '../data/place_list.pickle'
place_table_redu_file = '../data/place_list_redu.pickle'
place_table_redu2_file = '../data/place_list_redu2.pickle'
chi_table_file = '../results/chi_table.txt'

#drop_place_list = ['establishment', 'point_of_interest']
drop_place_list = []

import pickle
import numpy as np


def parse_dict(dict_name):

    fid = open(dict_name)
    line = fid.readline().decode('utf-8')
    emb_dict = {}
    while line:
        #line = line.encode('utf-8')
        word_seq = line.split(' ')
        emb_dict[word_seq[0]] = [float(elem) for elem in word_seq[1:]]

        line = fid.readline().decode('utf-8')

    fid.close()
    return emb_dict


def lookup_dict_placeType(tweets, dict_name):
    emb_dict = parse_dict(dict_name)
    emb_seq = []
    word_seq = []
    append_emp = False
    for elem in tweets:
        if elem['text']:
            if len(elem['text']) > 0:
                temp_seq = []
                temp_word_seq = []
                for word in elem['text']:
                    if emb_dict.has_key(word):
                        temp_seq.append(emb_dict[word])
                        temp_word_seq.append(word)

                if len(temp_seq) > 0:
                    emb_seq.append(temp_seq)
                    word_seq.append(temp_word_seq)
                else:
                    append_emp = True

            else:
                append_emp = True

        else:
            append_emp = True

        if append_emp:
            emb_seq.append([])
            word_seq.append([])
            append_emp = False

    return emb_seq, word_seq, emb_dict


def sample_words(word_list, prob_info, sample_count=SAMPLE_COUNT):
    out_word = []
    rand_max = prob_info['end'][-1] + 1
    rand_num = np.random.randint(0, rand_max, sample_count)
    for count in range(sample_count):
        start_arr = np.array(prob_info['start']) <= rand_num[count]
        end_arr = np.array(prob_info['end']) >= rand_num[count]

        sele_arr = np.logical_and(start_arr, end_arr)
        if np.sum(sele_arr) > 0:
            out_word.append(word_list[np.argwhere(sele_arr)[0][0]])

    return out_word



with open(twe_goo_place_file, 'r') as fid:
    twi_with_place, err_ind = pickle.load(fid)

place_type_map = {}
for elem, elem_ind in zip(twi_with_place, range(len(twi_with_place))):
    if elem.has_key('goo_result') and len(elem['text']) > 0:
        if elem['query_succ']:
            if len(elem['goo_result']) > 0:
                if elem['goo_result'][0].has_key('types'):
                    for place_type in elem['goo_result'][0]['types']:
                        if place_type not in drop_place_list:
                            if place_type_map.has_key(place_type):
                                place_type_map[place_type].append(elem_ind)
                            else:
                                place_type_map[place_type] = [elem_ind]

place_list = place_type_map.keys()
ind_list = place_type_map.values()
emb_seq, word_seq, emb_dict = lookup_dict_placeType(twi_with_place, emb_dict_file)

with open(place_table_file, 'w') as fid:
    pickle.dump(place_type_map, fid)

flat_word_list = {}
total_word_count = 0
for seq in word_seq:
    total_word_count += len(seq)
    for word in seq:
        if flat_word_list.has_key(word):
            flat_word_list[word] += 1
        else:
            flat_word_list[word] = 1

flat_word_prob = {}
for word in flat_word_list.keys():
    flat_word_prob[word] = flat_word_list[word]/float(total_word_count)

place_type_word = {}
place_word_count = {}
for addr_type in place_type_map.keys():
    word_list = {}
    local_count = 0
    for seq_ind in place_type_map[addr_type]:
        local_count += len(word_seq[seq_ind])
        for word in word_seq[seq_ind]:
            if word_list.has_key(word):
                word_list[word] += 1
            else:
                word_list[word] = 1

    place_type_word[addr_type] = word_list
    place_word_count[addr_type] = local_count

place_type_word_prob = {}
freq_word_key = flat_word_list.keys()
freq_word_count = flat_word_list.values()

freq_word_order = np.argsort(-1*np.array(freq_word_count))
freq_word_list = [freq_word_key[ind] for ind in freq_word_order[:TOP_FREQ_WORD]]

for addr_type in place_type_word.keys():
    local_word_list = []
    prob_list = []
    global_prob_list = []
    word_count = []
    prob_diff = []
    #chi_squ = []
    type_word_count = place_word_count[addr_type]
    for word in place_type_word[addr_type].keys():
        if place_type_word[addr_type][word] >= MIN_SUP and word not in freq_word_list:
            local_word_list.append(word)
            temp_count = place_type_word[addr_type][word]
            word_count.append(temp_count)
            temp_prob = temp_count/float(type_word_count)
            prob_list.append(temp_prob)
            temp_global_prob = flat_word_prob[word]
            global_prob_list.append(temp_global_prob)
            #prob_diff.append(temp_prob/temp_global_prob)
            prob_diff.append((temp_prob - temp_global_prob) ** 2/temp_global_prob)


    prob_diff_order = np.argsort(-1*np.array(prob_diff))
    local_word_list_out = [local_word_list[ind] for ind in prob_diff_order]
    prob_list_out = [prob_list[ind] for ind in prob_diff_order]
    global_prob_list_out = [global_prob_list[ind] for ind in prob_diff_order]
    word_count_out = [word_count[ind] for ind in prob_diff_order]
    prob_diff_out = [prob_diff[ind] for ind in prob_diff_order]
    type_info = {'word_list':local_word_list_out, 'prob_list':prob_list_out,
                 'global_prob':global_prob_list_out, 'word_count':word_count_out,
                 'prob_diff':prob_diff_out, 'type_word_count':type_word_count, 'chi-square':np.sum(prob_diff)}
    place_type_word_prob[addr_type] = type_info


with open(dist_phrase_file, 'w') as fid:
    for place in place_type_word_prob.keys():
        if len(place_type_word_prob[place]['word_list']) > 0:
            temp_ind = []
            for prob, ind in zip(place_type_word_prob[place]['prob_diff'], range(len(place_type_word_prob[place]['word_count']))):
                if prob > 1:
                    temp_ind.append(ind)

            #print >> fid, '{}  {}'.format(place, str(np.sum([place_type_word_prob[place]['prob_list'][ind] for ind in temp_ind])))
            print >> fid, '{}  {}'.format(place, place_type_word_prob[place]['chi-square'])
            print >> fid, '~~~~~~~~~~~~~~~~~~~'
            for ind in range(min(PRINT_WORD_COUNT, len(place_type_word_prob[place]['word_list']))):
                print >> fid, '{}: {}'.format(place_type_word_prob[place]['word_list'][ind],
                                             place_type_word_prob[place]['prob_diff'][ind])

            print >> fid, '\n'

place_chi_table = []
temp_place_list = []
for place in place_type_word_prob.keys():
    temp_place_list.append(place)
    place_chi_table.append(place_type_word_prob[place]['chi-square'])

chi_order = np.argsort(-1*np.array(place_chi_table))

with open(chi_table_file ,'w') as fid:
    for chi_ind in chi_order:
        print >> fid, '{}'.format(temp_place_list[chi_ind])

    print >> fid, '\n'
    for chi_ind in chi_order:
        print >> fid, '{}'.format(float(place_chi_table[chi_ind]))

place_redu_list = {}
for ind in range(len(temp_place_list)):
    if place_chi_table[ind] > 0:
        place_redu_list[temp_place_list[ind]] = place_type_word_prob[temp_place_list[ind]]['type_word_count']

with open(place_table_redu_file, 'w') as fid:
    pickle.dump(place_redu_list, fid)

place_redu_list = {}
for ind in range(len(temp_place_list)):
    if place_chi_table[ind] > 0.5:
        place_redu_list[temp_place_list[ind]] = place_type_word_prob[temp_place_list[ind]]['type_word_count']

with open(place_table_redu2_file, 'w') as fid:
    pickle.dump(place_redu_list, fid)

place_prob_table = {}
for place in place_type_word_prob.keys():
    prob_table_start = []
    prob_table_end = []
    start_count = 0
    for word_count in place_type_word_prob[place]['word_count']:
        prob_table_start.append(start_count)
        start_count += word_count
        prob_table_end.append(start_count - 1)

    prob_info = {'start':prob_table_start, 'end':prob_table_end}
    place_prob_table[place] = prob_info

place_sample = {}
for place in place_type_word_prob.keys():
    if len(place_type_word_prob[place]['word_count']) > 0:
        place_sample[place] = sample_words(place_type_word_prob[place]['word_list'], place_prob_table[place])
    else:
        place_sample[place] = []

flat_table_start = []
flat_table_end = []
start_count = 0
for word_count in freq_word_count:
    flat_table_start.append(start_count)
    start_count += word_count
    flat_table_end.append(start_count - 1)

flat_prob_info = {'start':flat_table_start, 'end':flat_table_end}
flat_sample = sample_words(freq_word_key, flat_prob_info)

place_std_vec = {}
place_mean_vec = {}
for place in place_sample.keys():
    temp_vec = np.zeros((1, EMB_DIM))
    for word in place_sample[place]:
        temp_vec = np.vstack((temp_vec, emb_dict[word]))

    if len(temp_vec) > 1:
        temp_vec = temp_vec[1:,:]
        _, vec_dim = temp_vec.shape
        std_vec = []
        mean_vec = []
        for ind in range(vec_dim):
            std_vec.append(np.std(temp_vec[:, ind]))
            mean_vec.append(np.mean(temp_vec[:, ind]))

        place_std_vec[place] = std_vec
        place_mean_vec[place] = mean_vec
        print '{}: {} - {}\n'.format(place, np.mean(std_vec), place_word_count[place])

    else:
        place_std_vec[place] = -1
        place_mean_vec[place] = -1
        print '{}: empty\n'.format(place)

temp_vec = np.zeros((1, EMB_DIM))
for word in flat_sample:
    temp_vec = np.vstack((temp_vec, emb_dict[word]))

temp_vec = temp_vec[1:,:]
_, vec_dim = temp_vec.shape
std_vec = []
flat_mean_vec = []
for ind in range(vec_dim):
    std_vec.append(np.std(temp_vec[:, ind]))
    flat_mean_vec.append(np.mean(temp_vec[:, ind]))

print 'Fullset: {} - {}\n'.format(np.mean(std_vec), total_word_count)
