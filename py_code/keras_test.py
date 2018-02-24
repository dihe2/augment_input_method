#!/usr/bin/python

LAYER_SIZE1 = 64
LAYER_SIZE2 = 64
DENSE_LAYER_SIZE = 64
GRAM_COUNT = 4
BATCH_SIZE = 256
EPOCHS = 3

RE_SAMPLE_CLASS_COUNT = 50

data_file_name = '/home/hd89cgm/cs512/fp/data/filtered_geo_ext.pickle'
dict_file_name = '/home/hd89cgm/cs512/fp/data/glove.twitter.27B.25d.txt'
out_dict_file_name = '/home/hd89cgm/cs512/fp/data/word_ind2_ext.pickle'
pre_train_weight_file = '../data/bi_lstm_25_64_64_1002.pickle'

import pickle
import numpy as np
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
from keras.layers.wrappers import Bidirectional
from numpy.random import permutation as randperm
from keras.layers import concatenate, Input, embeddings

frame_dim, vec_dim = 4, 25

from keras.layers import Input, LSTM, Dense
from keras.models import Model, Sequential

with open(pre_train_weight_file) as fid:
    pre_train_weights = pickle.load(fid)

# First, let's define a vision model using a Sequential model.
# This model will encode an image into a vector.
vision_model = Sequential()
vision_model.add(Bidirectional(LSTM(LAYER_SIZE1, dropout=0.2, recurrent_dropout=0.2, return_sequences=True), input_shape=(frame_dim, vec_dim)))
vision_model.add(Bidirectional(LSTM(LAYER_SIZE2, dropout=0.2, recurrent_dropout=0.2, return_sequences=False)))
vision_model.layers[0].set_weights(pre_train_weights[0])
vision_model.layers[1].set_weights(pre_train_weights[1])

# Now let's get a tensor with the output of our vision model:
image_input = Input(shape=(frame_dim, vec_dim), name='Word_Embedding')
encoded_image = vision_model(image_input)

# Next, let's define a language model to encode the question into a vector.
# Each question will be at most 100 word long,
# and we will index words as integers from 1 to 9999.
question_input = Input(shape=(94,), name='Location_Type_Input')
encoded_question = Dense(64, activation='tanh')(question_input)
#encode_drop = Dropout(0.2)(encoded_question)

# Let's concatenate the question vector and the image vector:
#merged = concatenate([encoded_question, encoded_image])
merged = concatenate([encoded_question, encoded_image])

den1 = Dense(DENSE_LAYER_SIZE, activation='tanh')(merged)
#drop1 = Dropout(0.2)(den1)

# And let's train a logistic regression over 1000 words on top:
output = Dense(1002, activation='softmax', name='One_Hot_Output')(den1)

# This is our final model:
vqa_model = Model(inputs=[image_input, question_input], outputs=output)
#vqa_model.layers[-1].set_weights(pre_train_weights[-1])
#temp_weights = vqa_model.layers[-3].get_weights()
#pre_weights = pre_train_weights[-3]
#pre_mean = np.mean(pre_weights[0])
#pre_std = np.std(pre_weights[0])

#temp_weights[0] = np.random.normal(pre_mean, pre_std, (len(temp_weights[0]), len(temp_weights[0][0])))
#for ind_counter in range(len(pre_weights[0])):
#    temp_weights[0][ind_counter] = pre_weights[0][ind_counter]

#temp_weights[1] = pre_weights[1]
#vqa_model.layers[-3].set_weights(temp_weights)

vqa_model.summary()

vqa_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

feats_train = np.random.uniform(-1, 1, (100, 4, 25))
labels_train_ind = np.random.randint(0, 1002, 100)
labels_train = np.zeros((100, 1002))
for sample, samp_ind in zip(labels_train_ind, range(100)):
    labels_train[samp_ind, sample] = 1.0

place_train_ind = np.random.randint(0, 94, 100)
place_feats_train = np.zeros((100, 94))
for sample, sample_ind in zip(place_train_ind, range(100)):
    place_feats_train[samp_ind, sample] = 1.0

history = vqa_model.fit([feats_train, place_feats_train], labels_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_data=([feats_train, place_feats_train], labels_train))
