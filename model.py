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

def read_IMDB_dataset():
	path_IMDB = 'aclImdb/train/'
	filenames = glob.glob(path_IMDB+'pos/*.txt')
	print(len(filenames))
	dataset = []
	maxLength = 0
	with progressbar.ProgressBar(len(filenames)) as bar:
		for fileIndex in range(len(filenames)):
			with open(filenames[fileIndex], 'r', encoding="utf-8") as file:
				data = file.read()
				dataset += data
				if(len(data) > maxLength):
					maxLength = len(data)
			bar.update(fileIndex)
	print(maxLength)

def __main__():
	### load dataset ###



	read_IMDB_dataset();
	
	print('we are done')

__main__()