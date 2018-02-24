#!/usr/bin/python

LAYER_SIZE1 = 256
LAYER_SIZE2 = 256
FULL_LAYER_SIZE = 256
GRAM_COUNT = 4
BATCH_SIZE = 256
EPOCHS = 10

RE_SAMPLE_CLASS_COUNT = 50

drop_place_list = ['establishment', 'point_of_interest']
#drop_place_list = []

data_file_name = '../data/geo_place_full.pickle'
dict_file_name = '../data/glove.twitter.27B.100d.txt'
out_dict_file_name = '../data/word_ind2_ext.pickle'
pre_trained_weights_file = '../data/bi_lstm_100_256_256_256_1002_no2.pickle'
place_list_file = '../data/place_list.pickle'

import pickle
import numpy as np
from sklearn.cross_validation import train_test_split
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
from keras.layers.wrappers import Bidirectional
from numpy.random import permutation as randperm
from keras.layers import Input, concatenate

out_dict_size = 0
place_list_size = 0

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


def lookup_dict_1hot_wplace(tweets, dict_name, out_dict_name, place_ind):
    emb_dict = parse_dict(dict_name)
    with open(out_dict_name, 'r') as fid:
        [word_to_ind, _] = pickle.load(fid)

    global out_dict_size
    out_dict_size = len(word_to_ind) + 1

    emb_seq = []
    word_ind_seq = []
    place_ind_list = []
    for elem in tweets:
        if elem['text'] and elem.has_key('goo_result'):
            if elem['query_succ']:
                if len(elem['text']) > 1 and len(elem['goo_result']) > 0:
                    if elem['goo_result'][0].has_key('types'):
                        temp_seq = []
                        temp_ind_seq = []
                        temp_place_list = []
                        for word in elem['text']:
                            if emb_dict.has_key(word):
                                temp_seq.append(emb_dict[word])
                                if word_to_ind.has_key(word):
                                    temp_ind_seq.append(word_to_ind[word])
                                else:
                                    temp_ind_seq.append(out_dict_size - 1)

                        if len(temp_seq) > 0:
                            emb_seq.append(temp_seq)
                            word_ind_seq.append(temp_ind_seq)
                            for place in elem['goo_result'][0]['types']:
                                if place not in drop_place_list:
                                    if place_ind.has_key(place):
                                        temp_place_list.append(place_ind[place])
                                    else:
                                        temp_place_list.append(-1)

                            place_ind_list.append(temp_place_list)

    return emb_seq, word_ind_seq, place_ind_list


def gen_seq_1hot_wplace(emb_seq, ind_seq, place_ind_list):

    global place_list_size
    vec_dim = len(emb_seq[0][0])
    feats_seq = []
    label_seq = []
    label_ind_seq = []
    place_ind_temp = []
    for elem, ind_elem, place_list_local in zip(emb_seq, ind_seq, place_ind_list):

        for word_ind in range(len(elem) - 1):
            temp_seq = np.zeros((GRAM_COUNT, vec_dim))
            for gram_counter in range(GRAM_COUNT):
                source_ind = word_ind - gram_counter
                tar_ind = GRAM_COUNT - 1 - gram_counter
                if source_ind >= 0:
                    temp_seq[tar_ind] = elem[source_ind]

            feats_seq.append(temp_seq)
            label_seq.append(elem[word_ind + 1])
            label_ind_seq.append(ind_elem[word_ind + 1])
            place_ind_temp.append(place_list_local)

    feats_out = np.zeros((len(label_seq), GRAM_COUNT, vec_dim))
    labels_out = np.zeros((len(label_seq), vec_dim))
    labels_1hot = np.zeros((len(label_seq), out_dict_size))
    place_feats = np.zeros((len(label_seq), place_list_size))
    for ind_counter in range(len(label_seq)):
        feats_out[ind_counter, :, :] = feats_seq[ind_counter]
        labels_out[ind_counter, :] = label_seq[ind_counter]
        labels_1hot[ind_counter, label_ind_seq[ind_counter]] = 1.0
        for place in place_ind_temp[ind_counter]:
            if place >= 0:
                place_feats[ind_counter, place] = 1.0

    return feats_out, labels_out, labels_1hot, np.array(label_ind_seq), place_feats


def balance_sampl(data_labels):
    labels_ind = []
    labels_count = []
    for label in range(out_dict_size):
        labels_ind.append(np.argwhere(data_labels == label).ravel())
        labels_count.append(np.sum(data_labels == label))

    #count_order = np.argsort(labels_count)
    count_mean = np.mean(labels_count)
    count_std = np.std(labels_count)

    up_bound = count_mean + count_std
    low_bound = count_mean - count_std
    if low_bound <= 0:
        low_bound = count_mean

    ind_out_temp = []
    for ind_list, count in zip(labels_ind, labels_count):
        if count > up_bound:
            ind_list = randperm(ind_list)
            ind_out_temp.append(ind_list[:np.int(up_bound)])

        elif count < low_bound and count > 0:
            rep_count = min(np.int(np.ceil(low_bound/float(count))), 3)
            for rep in range(rep_count):
                ind_list = np.hstack((ind_list, ind_list))

            ind_out_temp.append(ind_list)

        else:
            ind_out_temp.append(ind_list)

    ind_out = np.array([])
    for ind_list in ind_out_temp:
        ind_out = np.hstack((ind_out, ind_list))

    ind_out = [np.int(numb) for numb in ind_out]
    return randperm(ind_out)


with open(data_file_name, 'r') as fid:
    twi_data_place, _ = pickle.load(fid)

with open(place_list_file, 'r') as fid:
    place_map_list = pickle.load(fid)

place_list = place_map_list.keys()
place_list_size = len(place_list)
place_ind_map = {}
for place, place_ind in zip(place_list, range(place_list_size)):
    place_ind_map[place] = place_ind

emb_seq, ind_seq, place_ind_list = lookup_dict_1hot_wplace(twi_data_place, dict_file_name, out_dict_file_name, place_ind_map)
feats, labels, labels_1hot, labels_ind, place_feats = gen_seq_1hot_wplace(emb_seq, ind_seq, place_ind_list)

sel_labels = balance_sampl(labels_ind)

feats = feats[sel_labels, :, :]
labels_1hot = labels_1hot[sel_labels, :]
labels_ind = labels_ind[sel_labels]
place_feats = place_feats[sel_labels, :]

_, frame_dim, vec_dim = feats.shape
_, out_dim = labels_1hot.shape

for dim_counter in range(vec_dim):
    feats[:, :, dim_counter] = (feats[:, :, dim_counter] - np.mean(feats[:, :, dim_counter]))/np.std(feats[:, :, dim_counter])

train_ind, test_ind = train_test_split(range(len(labels_ind)), test_size = 0.1)

feats_train = feats[train_ind, :, :]
place_train = place_feats[train_ind, :]
labels_ind_train = labels_1hot[train_ind, :]

feats_test = feats[test_ind, :, :]
place_test = place_feats[test_ind, :]
labels_ind_test = labels_1hot[test_ind,:]

# Build network
with open(pre_trained_weights_file) as fid:
    pre_train_weights = pickle.load(fid)

seq_model = Sequential()
seq_model.add(Bidirectional(LSTM(LAYER_SIZE1, dropout=0.2, recurrent_dropout=0.2, return_sequences=True), input_shape=(frame_dim, vec_dim)))
seq_model.add(Bidirectional(LSTM(LAYER_SIZE2, dropout=0.2, recurrent_dropout=0.2, return_sequences=False)))
seq_model.layers[0].set_weights(pre_train_weights[0])
seq_model.layers[1].set_weights(pre_train_weights[1])

emb_input = Input(shape=(frame_dim, vec_dim))
lstm_sub = seq_model(emb_input)

place_input = Input(shape=(place_list_size,))

merged = concatenate([lstm_sub, place_input])

den1 = Dense(FULL_LAYER_SIZE, activation='tanh')(merged)
drop1 = Dropout(0.2)(den1)

output = Dense(out_dim, activation='softmax')(drop1)

# This is our final model:
full_model = Model(inputs=[emb_input, place_input], outputs=output)

full_model.layers[-1].set_weights(pre_train_weights[-1])
temp_weights = full_model.layers[-3].get_weights()
pre_weights = pre_train_weights[-3]
pre_mean = np.mean(pre_weights[0])
pre_std = np.std(pre_weights[0])

temp_weights[0] = np.random.normal(pre_mean, pre_std, (len(temp_weights[0]), len(temp_weights[0][0])))
for ind_counter in range(len(pre_weights[0])):
    temp_weights[0][ind_counter] = pre_weights[0][ind_counter]

temp_weights[1] = pre_weights[1]
full_model.layers[-3].set_weights(temp_weights)


full_model.summary()

full_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

history = full_model.fit([feats_train, place_train], labels_ind_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_data=([feats_test, place_test], labels_ind_test))

score = full_model.evaluate([feats_test, place_test], labels_ind_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
