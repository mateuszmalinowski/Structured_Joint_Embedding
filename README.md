# Structured Joint Embedding (SJE) for 0-shot learning
A Theano-based implementation of SJE for 0-shot learning [1].
It is an unofficial implementation of [1] and may not be reliable.
The source code has only an education purpose.

[1] Z. Akata et. al. "Zero-Shot Learning with Structured Embeddings"
link: http://arxiv.org/pdf/1409.8403v1.pdf

# 0-shot learning
In the 0-shot learning scenario, the training and test classes are 
disjoint. To facilitate recognition the model has to successfully transfer
additional information (for instance attributes) from known to unknown classes.
In this example first an image is mapped into its representation 
(input embedding) with a global feature extractor (e.g. CNN). Next, a
a class name is encoded into its class representation
(output embedding) using word2vec. Later, we train a compatibility
function such that 
'''
F(input\_embedding(image), output\_embedding(class)) 
'''
is large if '''image''' has class '''class'''.

# Structured Joint Embedding (SJE)
The objective function is a binary ranking loss that separates positive
compatibilities from negative compatibilities 
(following the ideas of structured SVM formulation).
A compatibility is a function that measures dissimilarity between
input and output embeddings. 

The compatibility function between x and y is expressed as xWy with
W being the compatibility matrix. This function is also  similar to 
Mahalanobis distance but without positive definite or even symmetric 
requirements for the compatibility matrix.
Thus the input and output embeddings can exhibit different dimensions.

# Experiments
I tested SJE on the CUB dataset with class word2vec output embedding,
and achieved test accuracy ~= 22% 
This result corresponds to Table1, SJE column, CNN row in CUB \phi^w in [1].

# Tested on
 * Python 2.7.3
 * Theano:167df2c43d1d08000105d448ff04b5bf2a6400c4
