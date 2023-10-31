#!/bin/bash

wget https://www.openslr.org/resources/60/train-clean-100.tar.gz
tar -xf --no-same-permissions train-clean-100.tar.gz
rm train-clean-100.tar.gz

#wget https://www.openslr.org/resources/60/test-clean.tar.gz
#tar -xf --no-same-permissions test-clean.tar.gz
#rm test-clean.tar.gz

find $(pwd)/LibriTTS/train-clean-100 -name *.wav > filelist.train
#find $(pwd)/LibriTTS/test-clean -name *.wav > filelist.val