import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from nltk.corpus import stopwords

from gensim.parsing.preprocessing import strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short, stem_text
from gensim.parsing.preprocessing import preprocess_string, preprocess_documents
from gensim.utils import simple_preprocess

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import os
import re
import string
import random
import glob
import argparse
import functools

def strip_short2(s):
	return strip_short(s, 2)

from words import CustomWords

custom = CustomWords()
STOCKS = custom.get_stock_symbols()

# Default
DATA_PATH = "reddit_wsb.csv"
START_TIME = '2021-01-28'
END_TIME = '2021-03-31'

# Alternative 
# DATA_PATH = "r_wallstreetbets_posts.csv"
# START_TIME = '2020-12-01'
# END_TIME = '2021-02-15'

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
	# docs = [[word for word in preprocess_string(doc, CUSTOM_FILTERS) if word not in STOP_WORDS] for doc in docs]
	
	# Preserve stock symbols
	docs = [[word for word in preprocess_string(doc, CUSTOM_FILTERS) if (word in STOCKS) or (word not in STOP_WORDS)] for doc in docs]
	return docs

class WsbData:
	def __init__(self, start=START_TIME, end=END_TIME, data_path=DATA_PATH):
		self.data_path = data_path
		if data_path == "reddit_wsb.csv":
			self.df = pd.read_csv(self.data_path, index_col=['timestamp'])
			self.df.index = pd.to_datetime(self.df.index)
			self.df = self.df.sort_values(by=['timestamp'])
			self.df = self.df[pd.Timestamp(start):pd.Timestamp(end)+pd.Timedelta(days=1)]
		elif data_path == "r_wallstreetbets_posts.csv":
			self.df = pd.read_csv(self.data_path, index_col=['created_utc'])
			self.df.index = pd.to_datetime(self.df.index.rename('timestamp'), unit='s')
			self.df = self.df.sort_values(by=['timestamp'])
			self.df = self.df.loc[self.df['removed_by'].isnull()] # Keep existing posts
			self.df = self.df[pd.Timestamp(start):pd.Timestamp(end)+pd.Timedelta(days=1)]

	def get_df(self, verbose=False):
		if verbose:
			print(self.df)
		# Raw data
		return self.df

	def get_documents(self, verbose=False):
		# Join text
		doc_df = (self.df['title'].fillna('') + " " + self.df['body'].fillna('')).to_frame()
		doc_df.columns = ['Doc']
		if verbose:
			print(doc_df)
		# Title and content
		return doc_df

	def get_titles(self, verbose=True):
		title_df = self.df['title'].fillna('').to_frame()
		title_df.columns = ['Doc']
		if verbose:
			print(title_df)
		# Title
		return title_df

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
	def __init__(self, stocks=['GME'], start=START_TIME, end=END_TIME, filldates=True):
		# Default check closing price of Gamestop
		tickers = '^GSPC' #'^SPX'
		for s in stocks:
			tickers = tickers + ' ' + s
		self.df = yf.download(tickers, start=pd.Timestamp(start)-pd.Timedelta(days=7), end=pd.Timestamp(end)+pd.Timedelta(days=7))['Adj Close']
		self.df = self.df.round(2)
		self.start = start
		self.end = end

		# Fill no-trading dates, forward and backward fill
		if filldates:
			idx = pd.date_range(start=pd.Timestamp(start)-pd.Timedelta(days=7), end=pd.Timestamp(end)+pd.Timedelta(days=7))
			self.df = self.df.reindex(idx)
			self.df.ffill(axis=0, inplace=True)
			self.df.bfill(axis=0, inplace=True)

	def get_df(self):
		# Adjusted closing price
		return self.df[pd.Timestamp(self.start):pd.Timestamp(self.end)]

	def get_daily_returns(self):
		daily_rets = (self.df / self.df.shift()) - 1
		daily_rets = daily_rets[pd.Timestamp(self.start):pd.Timestamp(self.end)].fillna(0.)
		# Daily return
		return daily_rets

	def get_three_day_returns(self):
		three_day_rets = (self.df / self.df.shift(5)) - 1
		three_day_rets = three_day_rets[pd.Timestamp(self.start):pd.Timestamp(self.end)].fillna(0.)
		# Three day return
		return three_day_rets

	def get_normalized(self):
		# normed_df = self.df/self.df.iloc[0]
		# normed_df = normed_df[pd.Timestamp(self.start):pd.Timestamp(self.end)]

		normed_df = self.df[pd.Timestamp(self.start):pd.Timestamp(self.end)]
		normed_df = normed_df/normed_df.iloc[0]

		# Log10
		normed_df = np.log10(normed_df)
		return normed_df


if __name__ == '__main__':
	wsb = WsbData()
	print(wsb.get_df())
	stock = StockData()
	print(stock.get_df())
	