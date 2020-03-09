#!/usr/bin/env bash

echo 'Preparing OAI data for training...'
python ../common/prepare_oai_ssl.py

echo 'Preparing the full OAI setting for SL model ...'
python ../common/prepare_oai_full_sl.py

echo 'Preparing MOST data for evaluation ...'
python ../common/prepare_most_eval.py