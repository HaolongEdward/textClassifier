import numpy as np
# import pandas as pd
import pickle
from collections import defaultdict
import re
# from bs4 import BeautifulSoup
import sys
import os
import glob
import errno
import progressbar

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint

def read_IMDB_dataset():
	path_IMDB = 'aclImdb/train/'
	rawDatas = []
	labels = []

	posFileNames = glob.glob(path_IMDB+'pos/*.txt')
	negFileNames = glob.glob(path_IMDB+'neg/*.txt')	
	for fileIndex in range(len(posFileNames)):
		with open(posFileNames[fileIndex], 'r', encoding="utf-8") as file:
			data = file.read()
			# the square brackets is important!!!!
			rawDatas += [data]
			labels += [0]
	for fileIndex in range(len(negFileNames)):
		with open(negFileNames[fileIndex], 'r', encoding="utf-8") as file:
			data = file.read()
			rawDatas += [data]
			labels += [1]
		
	return [rawDatas, labels]
	# print('Maximum length of the string is: ', MAX_SEQUENCE_LENGTH)

	
def preprocessing(rawDatas):

	tokenizer = Tokenizer(lower=True, char_level=True)
	tokenizer.fit_on_texts(rawDatas)
	sequences = tokenizer.texts_to_sequences(rawDatas)

	print('Number of unique tokens', tokenizer.word_index)

	MAX_SEQUENCE_LENGTH = len(max(sequences, key=len))
	print('the longest sequence in rawDatas is: ', MAX_SEQUENCE_LENGTH)
	# print('the length of a random sequence in rawDatas is: ', sequences[4500])

	dataset = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
	# print(dataset.shape)
	return dataset

def __main__():
	### load dataset ###
	rawdata_and_label = read_IMDB_dataset();
	
	### pre-processing ###
	data_set = preprocessing(rawdata_and_label[0])
	label_set = to_categorical(np.asarray(rawdata_and_label[1]))
	
	print('Shape of Data Tensor:', data_set.shape)
	print('Shape of Label Tensor:', label_set.shape)



	print('we are done')












__main__()






