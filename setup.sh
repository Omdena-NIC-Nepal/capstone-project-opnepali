#!/bin/bash
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt
python -m textblob.download_corpora
