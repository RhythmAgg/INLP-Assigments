import language_model as lm
import sys
import numpy as np
import tokenizer

lm_type = sys.argv[1]
corpus_path = sys.argv[2]
k = int(sys.argv[3])
ct = tokenizer.CustomTokenizer()
test_sentence = input("Input Sentence: ")
test_tokens = ct.generator_tokenizer(test_sentence)

if lm_type == 'i':
    params, li = lm.generator_call_i(corpus_path)
    for i in range(3):
        del lm.ngrams[i]['<UNK>']
    lm.corpus_size -= lm.ngrams[0][('<STR>',)]
    del lm.ngrams[0][('<STR>',)]

    vocab = list(lm.ngrams[0].keys())[:-1]

    for sentence in test_tokens:
        sentence = ['<STR>' for _ in range(2)] + sentence
        context = tuple(sentence[-2:])
        calc_prob = lambda N,d,D:  (N / D) if D > 0 else 0.0
        probs = []
        for word in vocab:
            gram = context + word
            prob = params[0]*calc_prob(lm.ngrams[2][gram], 1, lm.ngrams[1][gram[:-1]]) + params[1]*calc_prob(lm.ngrams[1][gram[1:]],0, lm.ngrams[0][gram[1:2]]) + params[2]*((lm.ngrams[0][gram[2:3]]) / lm.corpus_size)
            probs.append(prob)
        probs = np.array(probs)
        best_words_index = np.argsort(probs)[::-1]
        for i in best_words_index[:k]:
            print(vocab[i], probs[i])
        # print(len(lm.ngrams[0]))
elif lm_type == 'n':
    ngrams = lm.generator_call_n(corpus_path)
    vocab = list(ngrams[0].keys())[:-1]

    for sentence in test_tokens:
        sentence = ['<STR>' for _ in range(2)] + sentence
        context = tuple(sentence[-2:])
        calc_prob = lambda N,D:  (N / D) if D > 0 else 0.0
        probs = []
        for word in vocab:
            gram = context + word
            prob = calc_prob(ngrams[2][gram], ngrams[1][context])
            probs.append(prob)
        probs = np.array(probs)
        best_words_index = np.argsort(probs)[::-1]
        for i in best_words_index[:k]:
            print(vocab[i], probs[i])
        # print(len(lm.ngrams[0]))