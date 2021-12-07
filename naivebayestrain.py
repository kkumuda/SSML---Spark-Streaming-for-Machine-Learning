import re
import string

import numpy as np


    
def train_naive_bayes(freqs, train_x, train_y):
    
    loglikelihood = {}
    logprior = 0
    
    #  V =the number of unique words in the vocabulary
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)
    
    
    N_pos = N_neg = 0
    
    for pair in freqs.keys():
        
        if pair[1] > 0:
            
            N_pos += freqs.get(pair, 1)
        
        else:
            N_neg += freqs.get(pair, 1)
    
    D = len(train_y) #D=no of documents
    D_pos = sum(train_y)
    D_neg = D-D_pos

    logprior = np.log(D_pos)-np.log(D_neg)


    for word in vocab:

        freq_pos = freqs.get((word, 1),0)
        freq_neg = freqs.get((word, 0),0)
       
       #  probability each word is positive,negative
        p_w_pos = (freq_pos + 1)/(N_pos + V)
        p_w_neg = (freq_neg + 1)/(N_neg + V)


        loglikelihood[word] = np.log(p_w_pos/p_w_neg)
    return logprior, loglikelihood

logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)