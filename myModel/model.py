import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import re
# from bs4 import BeautifulSoup
import sys
import os

import errno
import progressbar
import keras.backend as K
import data_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, InputLayer, Flatten, LeakyReLU, Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.optimizers import Adam
# from keras.models import Model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
	
def IMDN_preprocessing(rawDatas):

	tokenizer = Tokenizer(lower=True, char_level=True)
	tokenizer.fit_on_texts(rawDatas)
	sequences = tokenizer.texts_to_sequences(rawDatas)
	num_unique_token = len(tokenizer.word_index)
	print('Number of unique tokens', num_unique_token)

	MAX_SEQUENCE_LENGTH = len(max(sequences, key=len))
	print('the longest sequence in rawDatas is: ', MAX_SEQUENCE_LENGTH)
	# print('the length of a random sequence in rawDatas is: ', sequences[4500])

	dataset = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

	print(dataset.shape)
	
	return dataset, num_unique_token

def IMDN_experiments():
	data_set, label_set = data_utils.read_IMDB_dataset();

### pre-processing ###
	data_set, num_unique_token = IMDN_preprocessing(data_set)
	label_set = to_categorical(np.asarray(label_set))
	
	print('Shape of Data Tensor:', data_set.shape)
	print('Shape of Label Tensor:', label_set.shape)

### build cnn model ###
	cnn = get_CNN_model(num_unique_token, data_set.shape[1], label_set.shape[1])

	learning_rate = 0.001
	batch_size = 128
	epochs = 10

###	compile model and set up experiments ###
	cnn.compile(
		optimizer=Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), 
		loss='categorical_crossentropy', 
		metrics=['accuracy']
		)
	cnn.summary()

	dataset_train, dataset_test, target_train, target_test = \
								train_test_split(data_set, label_set, test_size=0.1)
### experiment starts ###	
	cnn.fit(
		dataset_train, target_train,
		batch_size = batch_size,
		epochs = epochs,
		verbose=1,
		validation_data=(dataset_test, target_test)
	)
	score = cnn.evaluate(dataset_test, target_test, batch_size=batch_size)
	print(score)
	K.clear_session()

def get_CNN_model(num_unique_token, MAX_SEQUENCE_LENGTH, num_class):
	feature_dimension = 64
	cnn = Sequential()
	# cnn.add(InputLayer(MAX_SEQUENCE_LENGTH))

	cnn.add(Embedding(num_unique_token+1, feature_dimension, input_length=MAX_SEQUENCE_LENGTH))
	cnn.add(Conv1D(128, kernel_size=3))
	cnn.add(LeakyReLU(0.1))
	cnn.add(Dropout(rate=0.25))

	cnn.add(Conv1D(64, kernel_size=3))
	cnn.add(LeakyReLU(0.1))
	cnn.add(Dropout(rate=0.25))
	
	cnn.add(Conv1D(32, kernel_size=3))
	cnn.add(LeakyReLU(0.1))
	cnn.add(Dropout(rate=0.25))
	
	cnn.add(Flatten())

	cnn.add(Dense(128))
	cnn.add(LeakyReLU(0.1))
	cnn.add(Dropout(rate=0.25))

	cnn.add(Dense(16))
	cnn.add(LeakyReLU(0.1))
	cnn.add(Dropout(rate=0.25))
	
	cnn.add(Dense(num_class, activation='softmax'))
	# cnn.add(LeakyReLU(0.1))
	# cnn.add(Activation('softmax'))
	return cnn

def textInception(input, num_classes):
	net = Embedding(num_unique_token+1, feature_dimension, input_length=MAX_SEQUENCE_LENGTH)(input)
	return net

def inception_experiments():

	path_to_file = '../data/wiki_movie_plots/wiki_movie_plots_deduped.csv' 
	colnmaes = ['Release Year','Title','Origin/Ethnicity','Director','Cast','Genre','Wiki Page','Plot']
	data = pd.read_csv(path_to_file, names=colnmaes)

	print(data.Genre.tolist())



def __main__():
	### load dataset ###
	K.clear_session()

	inception_experiments()

	# IMDN_experiments()




	K.clear_session()

	print('we are done')












__main__()






