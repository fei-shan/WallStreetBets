import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

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

SYMBOLS = {'GME': 'Gamestop'
		   , 'AMC': 'AMC'
		   , 'NOK': 'Nokia'
		   , 'BB': 'Blackberry'
		   # , 'BBBY': 'Bed\sBath'
		   # , 'EXPR': 'Express'
		   # , 'KOSS': 'Koss'
		   # , 'NAKD': 'Naked\sBrand'
		  }
CUSTOM = CustomWords()

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="verbose")
# parser.add_argument("-p", "--plot", dest="plot", action="store_true", help="plot")
parser.add_argument("-t", "--title-only", dest="title", action="store_true", help="only check post title")
parser.add_argument("-m", "--mentioned-only", dest="mentioned", action="store_true", help="only check post that mentions the stock")
parser.add_argument("-g", "--gme-only", dest="gme", action="store_true", help="only check GME")
parser.add_argument("-w", "--weighted-sentiment", dest="weighted", action="store_true", help="Weight sentiment by post score")
parser.set_defaults(verbose=False, title=False, mentioned=False, gme=False)
args = parser.parse_args()

if args.gme:
	SYMBOLS = {'GME': 'Gamestop'}

def time_lagged_corr(s1, s2, lag_range=None):
	if not lag_range:
		lag_range = s2.size-1

	max_corr = float('-inf')
	min_corr = float('inf')
	max_lag = 0
	min_lag = 0

	# Shift s2
	for lag in range(-lag_range, lag_range+1):
		corr = s1.corr(s2.shift(-lag)) # delay
		if corr > max_corr:
			max_corr = corr
			max_lag = lag
		if corr < min_corr:
			min_corr = corr
			min_lag = lag

	return max_corr, max_lag, min_corr, min_lag

def generate_wordcloud(doc_df):
	text = ' '.join(doc for doc in doc_df['Doc'])
	stopwords = set(CUSTOM.get_git_stopwords())
	stopwords = stopwords.union(set(CUSTOM.get_more_stopwords()))
	wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

	plt.figure(figsize=[19.2, 10.8])
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.savefig("wordcloud")

def compare_sentiment_return(doc_df, stock_df, daily_rets, symbols=SYMBOLS):
	# Combine symbols
	valid_symbols = [s + '(?:\s|$)' + '|' + symbols[s] + '(?:\s|$)' for s in symbols.keys()] # make sure the word end with space
	valid_docs = doc_df
	if args.mentioned:
		# Check if the stock symbol exist in the doc
		valid_docs = valid_docs.loc[doc_df['Doc'].str.contains('|'.join(valid_symbols), case=False)] # combine stocks
		valid_docs = valid_docs.loc[~valid_docs['Doc'].str.contains('megathread', case=False)]

	print(valid_docs)
	
	# Get average sentiment
	avg_daily_scores = valid_docs[['Sentiment', 'Market Sentiment', 'Score']]
	if args.weighted:
		weights = avg_daily_scores['Score']/avg_daily_scores['Score'].sum(axis=0) # Weighted by post score
		print(weights)
		avg_daily_scores['Sentiment'] = avg_daily_scores['Sentiment'].multiply(weights)
		avg_daily_scores['Market Sentiment'] = avg_daily_scores['Market Sentiment'].multiply(weights)
		avg_daily_scores = avg_daily_scores.groupby(pd.Grouper(freq='D')).mean()[['Sentiment', 'Market Sentiment']]
	else:
		avg_daily_scores = avg_daily_scores[['Sentiment', 'Market Sentiment']].groupby(pd.Grouper(freq='D')).mean()
	avg_daily_scores = avg_daily_scores.fillna(0.)

	# print(avg_daily_scores)

	# Get average stock price and return
	avg_stock_df = stock_df[[s for s in symbols.keys()]]
	avg_daily_rets = daily_rets[[s for s in symbols.keys()]]
	avg_stock_df.insert(0, 'Stock Price', 0.)
	avg_daily_rets.insert(0, 'Daily Return', 0.)
	avg_stock_df['Stock Price'] = avg_stock_df.mean(axis=1)
	avg_daily_rets['Daily Return'] = avg_daily_rets.mean(axis=1)

	# avg_stock_df = stock_df[[s for s in symbols.keys()]].rename(columns={s: "Stock Price"})
	# avg_daily_rets = stock_df[[s for s in symbols.keys()]].rename(columns={s: "Daily Return"})

	result = pd.concat([avg_daily_scores,
						avg_stock_df[['Stock Price']], 
						avg_daily_rets[['Daily Return']]], axis=1)
	# print(result)

	# Store data
	result.to_csv("{}result_{}.csv".format("weighted_" if args.weighted else "", [s for s in symbols.keys()]))

	# Stocks
	print("Stocks selected: {}".format([s for s in symbols.keys()]))
	print("Mean: {} => {}".format(result['Sentiment'].mean(), result['Market Sentiment'].mean()))
	print("Variance: {} => {}".format(result['Sentiment'].var(), result['Market Sentiment'].var()))
	print("Standard deviation: {} => {}".format(result["Sentiment"].std(), result['Market Sentiment'].std()))

	# Check counts
	for s in symbols.keys():
		print("Stock mentioned count: {} => {}".format(s, doc_df[doc_df['Doc'].str.contains('|'.join([s + '(?:\s|$)', symbols[s] + '(?:\s|$)']), case=False)].size))

	# Jargons
	corr = avg_daily_scores['Sentiment'].corr(avg_daily_scores['Market Sentiment'])
	print("Pearson correlation of sentiment and market sentiment is {}".format(corr))

	# Stock price with jargons
	corr = avg_stock_df['Stock Price'].corr(avg_daily_scores['Market Sentiment'])
	print("Pearson correlation of price and sentiment is {} with market sentiment".format(corr))

	max_corr, max_lag, min_corr, min_lag = time_lagged_corr(avg_stock_df['Stock Price'], avg_daily_scores['Market Sentiment'], 10)
	print("Maximum time lagged correlation of price and sentiment is {} with lag {} with market sentiment".format(max_corr, max_lag))
	print("Minimum time lagged correlation of price and sentiment is {} with lag {} with market sentiment".format(min_corr, min_lag))

	# Daily returns with jargons
	corr = avg_daily_rets['Daily Return'].corr(avg_daily_scores['Market Sentiment'])
	print("Pearson correlation of return and sentiment is {} with market sentiment".format(corr))

	max_corr, max_lag, min_corr, min_lag = time_lagged_corr(avg_daily_rets['Daily Return'], avg_daily_scores['Market Sentiment'], 10)
	print("Maximum time lagged correlation of return and sentiment is {} with lag {} with market sentiment".format(max_corr, max_lag))
	print("Minimum time lagged correlation of return and sentiment is {} with lag {} with market sentiment".format(min_corr, min_lag))

	# Stock price without jargons
	corr = avg_stock_df['Stock Price'].corr(avg_daily_scores['Sentiment'])
	print("Pearson correlation of price and sentiment is {} without market sentiment".format(corr))

	max_corr, max_lag, min_corr, min_lag = time_lagged_corr(avg_stock_df['Stock Price'], avg_daily_scores['Sentiment'], 10)
	print("Maximum time lagged correlation of price and sentiment is {} with lag {} without market sentiment".format(max_corr, max_lag))
	print("Minimum time lagged correlation of price and sentiment is {} with lag {} without market sentiment".format(min_corr, min_lag))

	# Daily returns without jargons
	corr = avg_daily_rets['Daily Return'].corr(avg_daily_scores['Sentiment'])
	print("Pearson correlation of return and sentiment is {} without market sentiment".format(corr))

	max_corr, max_lag, min_corr, min_lag = time_lagged_corr(avg_daily_rets['Daily Return'], avg_daily_scores['Sentiment'], 10)
	print("Maximum time lagged correlation of return and sentiment is {} with lag {} without market sentiment".format(max_corr, max_lag))
	print("Minimum time lagged correlation of return and sentiment is {} with lag {} without market sentiment".format(min_corr, min_lag))

	# Plot
	ax = result[['Sentiment', 'Stock Price', 'Daily Return', 'Market Sentiment']].plot(kind='line', title='Average Sentiment vs. Stock Price/Daily Return').legend(loc='upper left')
	fig = ax.get_figure()
	fig.savefig("{}avg_sentiment_vs_stock_{}".format("weighted_" if args.weighted else "", valid_symbols), bbox_inches='tight')

	ax = result[['Stock Price', 'Market Sentiment']].plot(kind='line', title='Average Sentiment vs. Stock Price/Daily Return', color=['orange', 'red']).legend(loc='upper left')
	fig = ax.get_figure()
	fig.savefig("{}avg_sentiment_vs_stock_partial_{}".format("weighted_" if args.weighted else "", valid_symbols), bbox_inches='tight')

	dtw_distance_market, dtw_path_dtw = fastdtw(result['Stock Price'], result['Market Sentiment'], dist=euclidean)
	dtw_distance, dtw_path = fastdtw(result['Stock Price'], result['Sentiment'], dist=euclidean)
	print("Dynamic time warping price: {} => {}".format(dtw_distance, dtw_distance_market))
	dtw_distance_market, dtw_path_dtw = fastdtw(result['Daily Return'], result['Market Sentiment'], dist=euclidean)
	dtw_distance, dtw_path = fastdtw(result['Daily Return'], result['Sentiment'], dist=euclidean)
	print("Dynamic time warping daily return: {} => {}".format(dtw_distance, dtw_distance_market))


def main():
	# Default
	wsb_data = WsbData()
	stock_data = StockData(stocks=SYMBOLS)
	# Title and content
	doc_df = wsb_data.get_documents()
	
	# Title only with alternative dataset and time
	if args.title:
		wsb_data = WsbData(start='2020-12-01', end='2021-02-15', data_path="r_wallstreetbets_posts.csv")
		stock_data = StockData(start='2020-12-01', end='2021-02-15', stocks=SYMBOLS)
		doc_df = wsb_data.get_titles()

	doc_df['Doc'] = doc_df['Doc'].str.lower()

	# Sentiment score
	doc_df.insert(0, 'Sentiment', 0.) # Insert column
	doc_df.insert(0, 'Market Sentiment', 0.) # Insert column

	# With and without jargon
	sentiment_analyser = SentimentIntensityAnalyzer()
	market_sentiment_analyser = SentimentIntensityAnalyzer(lexicon_file="market_lexicon.txt")
	for index, row in doc_df.iterrows():
		doc_df.at[index, 'Sentiment'] = sentiment_analyser.polarity_scores(row['Doc'])['compound']
		doc_df.at[index, 'Market Sentiment'] = market_sentiment_analyser.polarity_scores(row['Doc'])['compound']

	# Returns
	daily_rets = stock_data.get_daily_returns()
	three_day_rets = stock_data.get_three_day_returns()

	# Adjusted closing prices
	stock_df = stock_data.get_df()

	# Normalized log prices
	normed_df = stock_data.get_normalized()

	# Check dataframe
	if args.verbose:
		print(doc_df)
		print(stock_df)
		print(daily_rets)

	# Analysis
	compare_sentiment_return(doc_df, normed_df, daily_rets, symbols=SYMBOLS)
	# generate_wordcloud(doc_df[['Doc']])

	return

if __name__ == '__main__':
	main()
