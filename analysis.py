import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

import gensim
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.wrappers import LdaMallet
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, preprocess_documents
from gensim.utils import simple_preprocess

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import spacy as sp
# !python3 -m spacy download en  # run in terminal once
import nltk

from words import CustomWords

import argparse



# Local
from dataset import WsbData, StockData

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="verbose")
	parser.add_argument("-p", "--plot", dest="verbose", action="store_true", help="plot")
	parser.set_defaults(verbose=False)
	args = parser.parse_args()

	verbose=args.verbose

	# Original
	wsb_data = WsbData()
	wsb_df = wsb_data.get_df()
	# wsb_docs = wsb.get_documents()
	# wsb_docs_tokenized = wsb.get_tokenized_documents()

	# Join text
	doc_df = wsb_df['title'].fillna('') + " " + wsb_df['body'].fillna('')

	# Sentiment score
	sentiment_analyser = SentimentIntensityAnalyzer()
	sentiment_scores = []
	for doc in doc_df:
		vs = sentiment_analyser.polarity_scores(doc)['compound']
		sentiment_scores.append(vs)
	score_df = pd.Series(sentiment_scores, index=doc_df.index)

	# Closing price
	stock_data = StockData()
	stock_df = stock_data.get_df()['GME']

	if verbose:
		for i in range(5):
			print("{}".format(doc_df[i]))
			print("{:*>65}".format(str(score_df[i])))
			print("{:*>65}".format(str(stock_df[i])))

	if plot:
		# TODO: plot
		pass

	return

if __name__ == '__main__':
	main()