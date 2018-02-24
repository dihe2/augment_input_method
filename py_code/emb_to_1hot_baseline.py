#!/usr/bin/python

LAYER_SIZE1 = 64
LAYER_SIZE2 = 64
GRAM_COUNT = 4
BATCH_SIZE = 256
EPOCHS = 5

data_file_name = '/home/hd89cgm/cs512/fp/data/filtered_geo_ext_2.pickle'
dict_file_name = '/home/hd89cgm/cs512/fp/data/glove.twitter.27B.25d.txt'
out_dict_file_name = '/home/hd89cgm/cs512/fp/data/word_ind2_ext.pickle'

import pickle
import numpy as np
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
from keras.layers.wrappers import Bidirectional

out_dict_size = 0


def parse_dict(dict_name):

    fid = open(dict_name)
    line = fid.readline().encode('utf-8')
    emb_dict = {}
    while line:
        #line = line.encode('utf-8')
        word_seq = line.split(' ')
        emb_dict[word_seq[0]] = [float(elem) for elem in word_seq[1:]]

        line = fid.readline().encode('utf-8')

    fid.close()
    return emb_dict


def lookup_dict_1hot(tweets, dict_name, out_dict_name):
    emb_dict = parse_dict(dict_name)
    with open(out_dict_name, 'r') as fid:
        [word_to_ind, ind_to_word] = pickle.load(fid)

    global out_dict_size
    out_dict_size = len(word_to_ind) + 1

    emb_seq = []
    word_ind_seq = []
    for elem in tweets:
        if elem['text']:
            if len(elem['text']) > 1:
                temp_seq = []
                temp_ind_seq = []
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

    return emb_seq, word_ind_seq


def gen_seq_1hot(emb_seq, ind_seq):

    vec_dim = len(emb_seq[0][0])
    feats_seq = []
    label_seq = []
    label_ind_seq = []
    for elem, ind_elem in zip(emb_seq, ind_seq):

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

    feats_out = np.zeros((len(label_seq), GRAM_COUNT, vec_dim))
    labels_out = np.zeros((len(label_seq), vec_dim))
    labels_ind_out = np.zeros((len(label_seq), out_dict_size))
    for ind_counter in range(len(label_seq)):
        feats_out[ind_counter, :, :] = feats_seq[ind_counter]
        labels_out[ind_counter, :] = label_seq[ind_counter]
        labels_ind_out[ind_counter, label_ind_seq[ind_counter]] = 1.0

    return feats_out, labels_out, labels_ind_out


with open(data_file_name) as fid:
    [twi_data_place, twi_data_coor] = pickle.load(fid)

emb_seq, ind_seq = lookup_dict_1hot(twi_data_place, dict_file_name, out_dict_file_name)
feats, labels, labels_ind = gen_seq_1hot(emb_seq, ind_seq)

_, frame_dim, vec_dim = feats.shape

for dim_counter in range(vec_dim):
    feats[:, :, dim_counter] = (feats[:, :, dim_counter] - np.mean(feats[:, :, dim_counter]))/np.std(feats[:, :, dim_counter])

train_ind, test_ind = train_test_split(range(len(labels)), test_size = 0.05)

feats_train = feats[train_ind,:,:]
labels_ind_train = labels_ind[train_ind,:]

feats_test = feats[test_ind,:,:]
labels_ind_test = labels_ind[test_ind,:]



model = Sequential()
model.add(Bidirectional(LSTM(LAYER_SIZE1, dropout=0.2, recurrent_dropout=0.2, return_sequences=True), input_shape=(frame_dim, vec_dim)))
model.add(Bidirectional(LSTM(LAYER_SIZE2, dropout=0.2, recurrent_dropout=0.2, return_sequences=False)))
model.add(Dense(out_dict_size, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

history = model.fit(feats_train, labels_ind_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_data=(feats_test, labels_ind_test))

score = model.evaluate(feats_test, labels_ind_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])