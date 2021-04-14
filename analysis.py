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

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import spacy as sp
# !python3 -m spacy download en  # run in terminal once
import nltk

from words import CustomWords

import argparse



# Local
from dataset import WsbData, StockData

SYMBOLS = ['GME']
CUSTOM = CustomWords()

def generate_wordcloud(doc_df):
	text = ' '.join(doc for doc in doc_df['Doc'])
	stopwords = set(CUSTOM.get_git_stopwords())
	stopwords = stopwords.union(set(CUSTOM.get_more_stopwords()))
	wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

	plt.figure(figsize=[19.2, 10.8])
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.savefig("wordcloud")

def compare_sentiment_return(doc_df, stock_df, symbols=['GME']):
	# Daily return of a stock
	daily_rets = (stock_df / stock_df.shift()) - 1
	daily_rets = daily_rets[:-1].fillna(0.)

	# Check if the stock symbol exist in the doc
	valid_docs = doc_df.loc[doc_df['Doc'].str.contains('|'.join(symbols), case=False)]
	print(valid_docs)
	# Get average sentiment
	daily_avg_scores = valid_docs['Sentiment'].groupby(pd.Grouper(freq='D')).mean()

	result = pd.concat([daily_rets, daily_avg_scores], axis=1)
	print(result)

	# Plot
	ax = result.plot(kind='line', title='Average Sentiment vs. Daily Return')
	fig = ax.get_figure()
	fig.savefig("avg_sentiment_vs_daily_return")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="verbose")
	parser.add_argument("-p", "--plot", dest="plot", action="store_true", help="plot")
	parser.set_defaults(verbose=False)
	args = parser.parse_args()

	verbose = args.verbose
	plot = args.plot

	# Original
	wsb_data = WsbData()
	wsb_df = wsb_data.get_df()
	# wsb_docs = wsb.get_documents()
	# wsb_docs_tokenized = wsb.get_tokenized_documents()

	# Join text
	doc_df = (wsb_df['title'].fillna('') + " " + wsb_df['body'].fillna('')).to_frame()
	doc_df.columns = ['Doc']

	# Sentiment score
	doc_df.insert(0, 'Sentiment', 0.) # Insert column
	sentiment_analyser = SentimentIntensityAnalyzer()
	for index, row in doc_df.iterrows():
		doc_df.at[index, 'Sentiment'] = sentiment_analyser.polarity_scores(row['Doc'])['compound']

	# sentiment_scores = []
	# for doc in doc_df['doc']:
	# 	vs = sentiment_analyser.polarity_scores(doc)['compound']
	# 	sentiment_scores.append(vs)
	# score_df = pd.Series(sentiment_scores, index=doc_df.index)

	# Adjusted closing price
	stock_data = StockData()
	stock_df = stock_data.get_df()[SYMBOLS]

	# Check dataframe
	if verbose:
		print(doc_df)
		print(stock_df)

	# Analysis
	compare_sentiment_return(doc_df, stock_df, SYMBOLS)
	generate_wordcloud(doc_df)

	if plot:
		# TODO: plot
		pass

	return

if __name__ == '__main__':
	main()