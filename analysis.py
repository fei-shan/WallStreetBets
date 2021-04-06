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

import argparse



# Local
from Dataset import WsbData

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-v",'--verbose', dest="verbose", action="store_true",
						help="verbose")
	parser.set_defaults(verbose=False)
	args = parser.parse_args()

	verbose=args.verbose

	wsb = WsbData()

	docs = wsb.get_documents()
	docs_tokenized = wsb.get_tokenized_documents()

	sentiment_analyser = SentimentIntensityAnalyzer()
	sentiment_scores=[]

	for doc in docs:
		vs = sentiment_analyser.polarity_scores(doc)
		
		sentiment_scores.append(vs)

	if verbose:
		print("{:-<65} {}".format(docs[0], str(sentiment_scores[0])))
	return

if __name__ == '__main__':
	main()