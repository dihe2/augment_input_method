#!/usr/bin/python

LAYER_SIZE1 = 128
LAYER_SIZE2 = 128
#LAYER_SIZE3 = 128
GRAM_COUNT = 4
BATCH_SIZE = 256
EPOCHS = 10
NNCOUNT = 3

data_file_name = '/home/hd89cgm/cs512/fp/data/filtered_geo_ext_2.pickle'
dict_file_name = '/home/hd89cgm/cs512/fp/data/glove.twitter.27B.100d.txt'

import pickle
import numpy as np
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from sklearn.neighbors import NearestNeighbors
from keras.layers.wrappers import Bidirectional


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


def lookup_dict(tweets, dict_name):
    emb_dict = parse_dict(dict_name)
    emb_seq = []
    word_seq = []
    for elem in tweets:
        if elem['text']:
            if len(elem['text']) > 1:
                temp_seq = []
                temp_word_seq = []
                for word in elem['text']:
                    if emb_dict.has_key(word):
                        temp_seq.append(emb_dict[word])
                        temp_word_seq.append(word)

                if len(temp_seq) > 0:
                    emb_seq.append(temp_seq)
                    word_seq.append(temp_word_seq)


    return emb_seq, word_seq


def gen_seq(emb_seq, word_seq):

    vec_dim = len(emb_seq[0][0])
    feats_seq = []
    label_seq = []
    word_ind_seq = []
    for elem, word in zip(emb_seq, word_seq):
        for word_ind in range(len(elem) - 1):
            temp_seq = np.zeros((GRAM_COUNT, vec_dim))
            for gram_counter in range(GRAM_COUNT):
                source_ind = word_ind - gram_counter
                tar_ind = GRAM_COUNT - 1 - gram_counter
                if source_ind >= 0:
                    temp_seq[tar_ind] = elem[source_ind]

            feats_seq.append(temp_seq)
            label_seq.append(elem[word_ind + 1])
            word_ind_seq.append(word[word_ind + 1])

    feats_out = np.zeros((len(label_seq), GRAM_COUNT, vec_dim))
    labels_out = np.zeros((len(label_seq), vec_dim))
    word_out = []
    for ind_counter in range(len(label_seq)):
        feats_out[ind_counter, :, :] = feats_seq[ind_counter]
        labels_out[ind_counter, :] = label_seq[ind_counter]
        word_out.append(word_ind_seq[ind_counter])

    return feats_out, labels_out, word_out


def gen_tree(dict_file, nn_count):

    emb_dict = parse_dict(dict_file)
    dict_key = emb_dict.keys()
    dict_feats = emb_dict.values()
    nbrs = NearestNeighbors(n_neighbors=nn_count, algorithm='ball_tree').fit(dict_feats)

    return nbrs, dict_key


with open(data_file_name) as fid:
    [twi_data_place, twi_data_coor] = pickle.load(fid)

emb_seq, word_seq = lookup_dict(twi_data_coor, dict_file_name)
feats, labels, labels_word = gen_seq(emb_seq, word_seq)
nn_tree, word_table = gen_tree(dict_file_name, NNCOUNT)

train_ind, test_ind = train_test_split(range(len(labels)), test_size = 0.001)

feats_train = feats[train_ind,:,:]
labels_train = labels[train_ind,:]

feats_test = feats[test_ind,:,:]
labels_test = labels[test_ind,:]
labels_word_test = [labels_word[temp_ind] for temp_ind in test_ind]

_, frame_dim, vec_dim = feats_train.shape

model = Sequential()
model.add(Bidirectional(LSTM(LAYER_SIZE1, dropout=0.2, recurrent_dropout=0.2, return_sequences=True), input_shape=(frame_dim, vec_dim)))
model.add(Bidirectional(LSTM(LAYER_SIZE2, dropout=0.2, recurrent_dropout=0.2, return_sequences=False)))
#model.add(Bidirectional(LSTM(LAYER_SIZE3, dropout=0.2, recurrent_dropout=0.2, return_sequences=False)))
model.add(Dense(vec_dim, activation='linear'))

model.summary()

model.compile(loss="mean_squared_error", optimizer="rmsprop")

history = model.fit(feats_train, labels_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_data=(feats_test, labels_test))
#score = model.evaluate(feats_test, labels_test, verbose=1)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

pred_out = model.predict(feats_test)
_, word_ind = nn_tree.kneighbors(pred_out)


word_out = []
for elem in word_ind:
    temp_word_list = [word_table[ind] for ind in elem]
    word_out.append(temp_word_list)

acc_count = 0
acc3_count = 0
for label, elem in zip(labels_word_test, word_out):
    if label.__eq__(elem[0].encode('utf-8')):
        acc_count += 1

    for sub_elem in elem:
        if label.__eq__(sub_elem.encode('utf-8')):
            acc3_count += 1
            break

print 'Top 1 acc rate: {}'.format(acc_count/float(len(labels_word_test)))
print 'Top 3 acc rate: {}'.format(acc3_count/float(len(labels_word_test)))

#acc_rate = acc_count/float(len(labels_word_test))
#acc3_rate = acc3_count/float(len(labels_word_test))
#
#with open('../results/corr_2_4gram_100_100_bi_10ep.pickle', 'w') as fid:
#   pickle.dump([model, labels_word_test, word_out, acc_rate, acc3_rate], fid)