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

def strip_short2(s):
	return strip_short(s, 2)

# Filter
# strip_short2 = functools.partial(strip_short, 2)
CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, 
	strip_numeric, remove_stopwords, 
	# strip_short, 
	stem_text]

# Remove more stopwords
STOP_WORDS = stopwords.words('english')
STOP_WORDS.extend(['from', 'subject', 're', 'edu', 'use', 'would', 'say', 'could', 'be', 
	'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 
	'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 
	'line', 'even', 'also', 'may', 'take', 'come', 'http', 'www'])


def custome_preprocessing(docs):
	docs = [[word for word in preprocess_string(doc, CUSTOM_FILTERS) if word not in STOP_WORDS] for doc in docs]
	return docs

class WsbData:
	def __init__(self):
		self.data_path = "reddit_wsb.csv"
		self.df = pd.read_csv(self.data_path, index_col=['timestamp'])
		self.df = self.df.sort_values(by=['timestamp'])

	def get_df(self, verbose=False):
		if verbose:
			print(self.df)
		return self.df

	def get_documents(self, verbose=False):
		# indices = self.df['title'].notnull()
		# text_df = self.df[indices]['title'] + " " + self.df[indices]['body'].fillna('')
		text_df = self.df['title'].fillna('') + " " + self.df['body'].fillna('')
		docs = text_df.tolist()
		return docs

	def get_tokenized_documents(self, verbose=False):
		# indices = self.df['title'].notnull()
		# text_df = self.df[indices]['title'] + " " + self.df[indices]['body'].fillna('')
		text_df = self.df['title'].fillna('') + " " + self.df['body'].fillna('')
		docs = text_df.tolist()
		if verbose:
			print(docs[0])
		docs_tokenized = custome_preprocessing(docs)
		# docs_tokenized = preprocess_documents(docs)
		if verbose:
			print(docs_tokenized[0])
		return docs_tokenized

class StockData():
	def __init__(self):
		self.df = None

if __name__ == '__main__':
	wsb = WsbData()
	print(wsb.get_df())
