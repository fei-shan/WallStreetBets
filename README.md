# WallStreetBets

Datasets available [here](https://www.kaggle.com/gpreda/reddit-wallstreetsbets-posts).
Alternative dataset under testing [here](https://www.kaggle.com/unanimad/reddit-rwallstreetbets).

Stock prices can be downloaded with yfinance Python library


## Files:

* **dataset.py**
For loading wsb posts and stock data

* **words.py**
Define custom stop words, stock symbols and market lexicon

* **analysis.py**
Run data analysis

## Run
Run 
```shell
python3 dataset.py
``` 
to generate the market jargons lexicon.

Copy and paste the original vader_lexicon.txt file in the VADER root folder, rename the copy as market_lexicon.txt, and append new generated market jargons to the file.

In the project root folder, run
```shell
python3 analysis.py
``` 