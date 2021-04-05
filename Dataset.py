import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from gensim.parsing.preprocessing import strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short, stem_text
from gensim.parsing.preprocessing import preprocess_string, preprocess_documents
from gensim.utils import simple_preprocess

import os
import re
import string
import random
import glob
import argparse
import functools

from nltk.corpus import stopwords

def strip_short2(s):
	return strip_short(s, 2)

# Filter
# strip_short2 = functools.partial(strip_short, 2)
CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, 
	strip_numeric, remove_stopwords, strip_short, stem_text]

# Remove more stopwords
STOP_WORDS = stopwords.words('english')
STOP_WORDS.extend(['from', 'subject', 're', 'edu', 'use', 'would', 'say', 'could', 'be', 
	'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 
	'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 
	'line', 'even', 'also', 'may', 'take', 'come', 'http', 'www'])


def custome_preprocessing(docs):

	docs = [[word for word in preprocess_string(doc, CUSTOM_FILTERS) if word not in STOP_WORDS] for doc in docs]
	return docs

class WsbDataset:
	def __init__(self):
		self.data_path = "reddit_wsb.csv"
		self.df = pd.read_csv(self.data_path, index_col=None)

	def get_df(self, verbose=False):
		if verbose:
			print(self.df)
		return self.df

	def get_documents(self, timestamp=False, verbose=False):
		indices = self.df['title'].notnull()
		# columns = ['title', 'body']
		# if timestamp:
		# 	columns.append('timestamp')

		text_df = self.df[indices]['title'] + " " + self.df[indices]['body'].fillna('')
		docs = text_df.tolist()
		return docs
		# docs = self.df[indices][columns].tolist()

	def get_tokenized_documents(self, timestamp=False, verbose=False):
		indices = self.df['title'].notnull()
		# columns = ['title', 'body']
		# docs = self.df[indices][columns].tolist()
		
		text_df = self.df[indices]['title'] + " " + self.df[indices]['body'].fillna('')
		docs = text_df.tolist()
		if verbose:
			print(docs[0])
		docs_tokenized = custome_preprocessing(docs)
		# docs_tokenized = preprocess_documents(docs)
		if verbose:
			print(docs_tokenized[0])
		return docs_tokenized

if __name__ == '__main__':
	wsb = WsbDataset()
	print(wsb.get_df())
