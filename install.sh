#!/bin/bash

pip3 install -r requirements.txt
python -m spacy download en
./weights/download_weights.sh
