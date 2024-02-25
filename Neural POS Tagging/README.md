# POS Tagger

## Contents
- pos_tagger.py
- FFN.ipynb
- RNN.ipynb
- ffn, RNN, GRU, LSTM folder
- report.pdf

### pos_tagger.py
Code for running the best performing models on an input sentence

```
python3 pos_tagger.py [-f|-r|-g|-l]
    -f: FFN
    -r: RNN
    -g: GRU
    -l: LSTM
```
Requires saved models in the same direcory structure as well as the training data conllu file for building the vocab

**Note**: Words unknown in the training corpus are replaced by the UNKNOW_TOKEN. Also words which have target POS tag not in the training target POS tags list are not counted in the dataset and while calculating loss, but they do come as context words.

### FFN.ipynb
Containg the training, validation and testing code for the FFN model. It contains the graphs, training and validation accuracies and process. Used to generate models saved in ./ffn 

### RNN.ipynb
Containg the training, validation and testing code for the RNN | GRU | LSTM models. It contains the graphs, training and validation accuracies and process. Used to generate models saved in ./RNN, ./GRU, ./LSTM

### Saved models
clone this repository in the root folder
```
git clone https://github.com/RhythmAgg/INLP-saved_models-2
cp -r ./INLP-saved_models-2/ .
```
Models saved following a naming convention described in the correspoding .ipynb files. These models are imported in the pos_tagger.py and used for inference.