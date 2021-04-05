import gensim
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.wrappers import LdaMallet
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, preprocess_documents
from gensim.utils import simple_preprocess

import spacy as sp
# !python3 -m spacy download en  # run in terminal once
import nltk
import vaderSentiment

import argparse



# Local
from Dataset import WsbDataset

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-v",'--verbose', dest="verbose", action="store_true",
						help="verbose")
	parser.set_defaults(training=False)
	args = parser.parse_args()

	wsb = WsbDataset()

	docs = wsb.get_tokenized_documents(verbose=True)


	return


if __name__ == '__main__':
	main()