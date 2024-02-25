import sys
import tokenizer
import numpy as np
import random
import math
from collections import defaultdict
import matplotlib.pyplot as plt
# from scipy.stats import linregress

def generate_ngrams(n):
    n_grams = defaultdict(int)
    start_token = ['<STR>' for _ in range(2)]
    for sentence in word_tokens:
        sentence = start_token+sentence
        for i in range(len(sentence) - n + 1):
            gram = tuple(sentence[i:i+n])
            n_grams[gram] += 1
    
    for freq in list(n_grams.values()):
        if freq == 1:
            n_grams['<UNK>'] += 1
    return n_grams

class LinearInterpolation:
    def parameters_setting(self):
        lambdas = [0.0 for _ in range(3)]
        for three_grams in ngrams[-1]:
            if(ngrams[2][three_grams] > 0):
                case1 = (ngrams[2][three_grams] - 1) / (ngrams[1][three_grams[:-1]] - 1) if ngrams[1][three_grams[:-1]] > 1 else 0
                case2 = (ngrams[1][three_grams[:-1]] - 1) / (ngrams[0][three_grams[:-2]] - 1) if ngrams[0][three_grams[:-2]] > 1 else 0
                case3 = (ngrams[0][three_grams[:-2]] - 1) / (corpus_size-1)

                max_case = max([case1, case2, case3])

                if(case1 == max_case):
                    lambdas[0] += ngrams[2][three_grams]
                elif(case2 == max_case):
                    lambdas[1] += ngrams[2][three_grams]
                else:
                    lambdas[2] += ngrams[2][three_grams]
        
        return np.array(lambdas) / np.sum(lambdas)

    def test_scores(self):
        scores = []
        # calc_prob = lambda N,d,D:  N / D if D > 0 else N / ngrams[d]['<UNK>']
        calc_prob = lambda N,d,D:  (N / D) if D > 0 else 0.0
        for sentence in test_tokens:
            sentence = ['<STR>' for _ in range(2)] + sentence
            score = 1.0
            for i in range(2,len(sentence)):
                gram = tuple(sentence[i-2:i+1])
                # print(ngrams[0][gram[2:3]], (ngrams[0][gram[2:3]] if ngrams[0][gram[2:3]] > 1 else ngrams[0]['<UNK>']) / corpus_size)
                prob = params[0]*calc_prob(ngrams[2][gram], 1, ngrams[1][gram[:-1]]) + params[1]*calc_prob(ngrams[1][gram[1:]],0, ngrams[0][gram[1:2]]) + params[2]*((ngrams[0][gram[2:3]] if ngrams[0][gram[2:3]] > 1 else ngrams[0]['<UNK>']) / corpus_size)
                score *= prob
            scores.append((max(score, 1e-200), len(sentence)-2))

        return scores

    def perplexity(self):
        perplexities = []
        for score, length in scores:
            h = -math.log2(score) / length
            perplexities.append(2**h)
        return perplexities
    
    def prob_sentence(self, text):
        sentence_tokens = tokenizer.tokenise(text, file = False)
        sentence = ['<STR>' for _ in range(2)] + sentence_tokens[0]
        score = 1.0
        calc_prob = lambda N,d,D:  (N / D) if D > 0 else 0.0
        for i in range(2,len(sentence)):
            gram = tuple(sentence[i-2:i+1])
            # print(ngrams[0][gram[2:3]], (ngrams[0][gram[2:3]] if ngrams[0][gram[2:3]] > 1 else ngrams[0]['<UNK>']) / corpus_size)
            prob = params[0]*calc_prob(ngrams[2][gram], 1, ngrams[1][gram[:-1]]) + params[1]*calc_prob(ngrams[1][gram[1:]],0, ngrams[0][gram[1:2]]) + params[2]*((ngrams[0][gram[2:3]] if ngrams[0][gram[2:3]] > 1 else ngrams[0]['<UNK>']) / corpus_size)
            score *= prob
        return score

class GoodTuring:
    def Nr(self):
        self.nr = defaultdict(int)
        for freq in ngrams[0].values():
            self.nr[freq] += 1
    
    def fit(self):
        zr = sorted(self.nr.items())[:-1]
        q = -1
        t = 1
        smoothed_nr = []
        for index, tup in enumerate(zr):
            smoothed_nr.append([tup[0], tup[1] / (0.5*(zr[t][0]-zr[max(q,0)][0]))])
            q += 1 
            t = min(len(zr)-1, t+1)
        smoothed_nr[-1][1] *= 0.5
        smoothed_nr = np.array(smoothed_nr)
        
        target = np.log(smoothed_nr[:,1])
        feature = np.log(smoothed_nr[:,0])
        self.slope, self.intercept, *_ = linregress(feature, target)
        self.predicted_zr = self.slope*feature + self.intercept
        
        plt.plot(feature, self.predicted_zr)
        plt.scatter(feature, target)
        plt.show()
        
    def new_count(self):
        self.new_counts = defaultdict(int)
        for freq in ngrams[0].values():
            self.new_counts[freq] = freq*((1 + 1/freq)**(self.slope + 1))
#             self.new_counts[freq] = (freq+1)*self.predicted_zr[freq+1] / self

    def bigram_counts(self):
        self.context_counts = defaultdict(int)
        self.context_unseen_vocab = defaultdict(int)
        self.context_unseen_vocab.default_factory = lambda: vocab
        for gram in ngrams[0].keys():
            context = gram[:-1]
            self.context_counts[context] += self.new_counts[ngrams[0][gram]]
            self.context_unseen_vocab[context] -= 1
            
        
    def test_scores(self):
        scores = []
        for sentence in test_tokens:
            sentence = ['<STR>' for _ in range(2)] + sentence
            score = 1.0
            for i in range(2,len(sentence)):
                gram = tuple(sentence[i-2:i+1])
                if(ngrams[0][gram] == 0):
                    prob = (np.exp(self.intercept) / (self.context_counts[gram[:-1]] + np.exp(self.intercept))) if self.context_counts[gram[:-1]] > 0 else 1e-5
                else:
                    prob = (self.new_counts[ngrams[0][gram]] / (self.context_counts[gram[:-1]] + np.exp(self.intercept)))
                score *= prob
            scores.append((max(score, 1e-200), len(sentence)))

        return scores
                                          
    def perplexity(self):
        perplexities = []
        avg_length = 0
        for score, length in scores:
            h = -math.log2(score) / length
            perplexities.append(2**h)
            avg_length += length
        return perplexities, avg_length / len(scores)
    
    def prob_sentence(self, text):
        sentence_tokens = tokenizer.tokenise(text, file = False)
        sentence = ['<STR>' for _ in range(2)] + sentence_tokens[0]
        score = 1.0
        for i in range(2,len(sentence)):
            gram = tuple(sentence[i-2:i+1])
            if(ngrams[0][gram] == 0):
                prob = (np.exp(self.intercept) / (self.context_counts[gram[:-1]] + np.exp(self.intercept))) if self.context_counts[gram[:-1]] > 0 else 1e-5
            else:
                prob = (self.new_counts[ngrams[0][gram]] / (self.context_counts[gram[:-1]] + np.exp(self.intercept)))
            score *= prob
        return score
        

def generator_call_i(corpus_path):
    global ngrams
    global word_tokens
    global corpus_size
    ngrams = []
    word_tokens = tokenizer.tokenise(corpus_path)
    li = LinearInterpolation()
    for n in range(3):
        ngrams.append(generate_ngrams(n+1))
    corpus_size = sum(ngrams[0].values()) - ngrams[0]['<UNK>']

    params = li.parameters_setting()
    return params, li

def generator_call_n(corpus_path):
    global ngrams
    global word_tokens
    global corpus_size
    ngrams = []
    word_tokens = tokenizer.tokenise(corpus_path)
    for n in range(3):
        ngrams.append(generate_ngrams(n+1))

    return ngrams

if __name__ == '__main__':
    random.seed(42)

    lm_type = sys.argv[1]
    corpus_path = sys.argv[2]
    tokens = tokenizer.tokenise(corpus_path)

    random.shuffle(tokens)

    split = len(tokens) - 1000

    word_tokens = tokens[:split]
    # word_tokens = tokens
    # test_tokens = tokens

    test_tokens = tokens[split:]
    # test_tokens = word_tokens

    ngrams = []

    if lm_type == 'i':
        li = LinearInterpolation()
        for n in range(3):
            ngrams.append(generate_ngrams(n+1))

        corpus_size = sum(ngrams[0].values()) - ngrams[0]['<UNK>']

        params = li.parameters_setting()

        # input_sentence = input("Input sentence: ")

        # print("Score: ", li.prob_sentence(input_sentence))

        scores = li.test_scores()

        perplexities = li.perplexity()
        with open('./Results/2021101081_LM4_test-perplexity.txt', "w") as f:
            f.write(f"{np.mean(perplexities)}\n")
            for i, sentence in enumerate(test_tokens):
                f.write(f"{sentence}    {perplexities[i]}\n")
        
        test_tokens = word_tokens
        scores = li.test_scores()

        perplexities = li.perplexity()
        with open('./Results/2021101081_LM4_train-perplexity.txt', "w") as f:
            f.write(f"{np.mean(perplexities)}\n")
            for i, sentence in enumerate(test_tokens):
                f.write(f"{sentence}    {perplexities[i]}\n")

    else:
        ngrams = []
        ngrams.append(generate_ngrams(3))
        ngrams.append(generate_ngrams(2))
        vocab = len(generate_ngrams(1))
        corpus_size = sum(ngrams[0].values()) - ngrams[0]['<UNK>']
        gt = GoodTuring()
        gt.Nr()
        gt.fit()
        gt.new_count()
        gt.bigram_counts()
        input_sentence = input("Input sentence: ")

        print("Score: ", gt.prob_sentence(input_sentence))
        # scores = gt.test_scores()
        # perplexities, avg_length = gt.perplexity()
        # with open('./Results/2021101081_LM3_test-perplexity.txt', "w") as f:
        #     f.write(f"{np.mean(perplexities)}\n")
        #     for i, sentence in enumerate(test_tokens):
        #         f.write(f"{sentence}    {perplexities[i]}\n")
        
        # test_tokens = word_tokens
        # scores = gt.test_scores()

        # perplexities, avg_length = gt.perplexity()
        # with open('./Results/2021101081_LM3_train-perplexity.txt', "w") as f:
        #     f.write(f"{np.mean(perplexities)}\n")
        #     for i, sentence in enumerate(test_tokens):
        #         f.write(f"{sentence}    {perplexities[i]}\n")





