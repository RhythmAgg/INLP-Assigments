# Language Modelling and Tokenization

## File Structure

The assigment submission contains the following files:
- `tokenizer.py` - Code for tokenization of the corpus
- `language_model.py` - Code for interpolation and good Turing models as well as generation of ngrams
- `Generator.py` - Code for generation of new words given a model and an input sentence

## Instructions for Each file
### tokenizer.py
```
python3 tokenizer.py
```
It asks for an input text and outputs the tokenized sentences. When imported in other files,The tokenizer functionality can be invoked through `tokenize` method
### language_model.py
```
python3 language_model.py <lm_type> <corpus_path>
```
**Command Line Arguements**
1. *lm_type*: `i` for interpolation, `g` for good Turing
2. *corpus_path*: File path of the Corpus

It asks for the input sequence and outputs the probability score for that input based on the model and the corpus provided. It was used to generate the **Results** by dividing the corpus into train and test sentences and then running inferences on the **test** split
### generator.py
```
python3 generator.py <lm_type> <corpus_path> <k>
```
**Command Line Arguements**
1. *lm_type*: `i` for interpolation, `n` for Ngrams
2. *corpus_path*: File path of the Corpus
3. *k*: No. of Candidates for the next word

It asks for the input sentence and output the `k` best next words sorted by their probability scores. It invokes functionalities from `language_model.py` and iterate over the *corpus* vocab to generate new words.