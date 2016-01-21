#!/usr/bin/env python
"""
Implementation of Structured Joint Embedding [1].
This code uses decreasing learning rate.

[1] Z. Akata et. al. "Zero-Shot Learning with Structured Embeddings"

Author: Mateusz Malinowski
Email: mmalinow@mpi-inf.mpg.de
"""
__docformat__ = 'restructedtext en'

import os
import sys
import time

import numpy

import theano
import theano.tensor as T

from load_data import load_data


class StructuredJointEmbedding(object):
    """
    Multi-class Structured SVM
    """

    def __init__(self, C, input_embedding, output_embedding, 
            n_input, n_output, batch_size, n_classes):
        """ 
        Initialize the parameters of the multiclass structured SVM.

        :type C: int
        :param C: data fidelity hyperparameter

        :type input_embedding: theano.tensor.TensorType
        :param input_embedding: symbolic variable that describes 
                      the input embedding of the
                      architecture (one minibatch);
                      input_embedding is in R[#data,n_input]

        :type output_embedding: theano.tensor.TensorType
        :param output_embedding: symbolic variable that described
                      the output embedding
                      output_embedding is in R[#classes,n_output]

        :type n_input: int
        :param n_input: input_embedding dimension

        :type n_output: int
        :param n_output: output_embedding dimension
        
        :type batch_size: int
        :param batch_size: batch size

        :type n_classes: int
        :param n_classes: number of classes
        """

        self.n_in = n_input
        self.n_out = n_output
        self.n_data = batch_size 
        self.n_classes = n_classes
        self.C = C
        self.input_embedding = input_embedding
        self.output_embedding = output_embedding

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        #self.W = theano.shared(
            #value=numpy.zeros((n_input, n_output), dtype=theano.config.floatX),
            #name='W', 
            #borrow=True)
        self.W = theano.shared(
            value=numpy.random.randn(n_input, n_output).astype(dtype=theano.config.floatX),
            name='W', 
            borrow=True)

        self.define_model()


    def define_compatibility(self):
         # in R[#data,n_output]
        self.compatibility_left = T.dot(self.input_embedding,self.W)
        # in R[#data,#classes]
        self.compatibility = T.dot(self.compatibility_left,self.output_embedding.T)


    def define_regularizer(self):
        self.l2norm = T.sum(self.W**2)


    def define_predictions(self):
        self.y_pred = T.argmax(self.compatibility, axis=1)


    def define_model(self):
        self.define_regularizer()
        self.define_compatibility()
        self.define_predictions()
        self.params = [self.W]
    

    def negative_log_likelihood(self, label_sym):
        """
        Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        :type label_sym: theano.tensor.TensorType
        :param label_sym: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # label_sym.shape[0] is (symbolically) the number of rows in label_sym, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(label_sym.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(label_sym.shape[0]),label_sym] is a vector
        # v containing [LP[0,label_sym[0]], LP[1,label_sym[1]], LP[2,label_sym[2]], ...,
        # LP[n-1,label_sym[n-1]]] and T.mean(LP[T.arange(label_sym.shape[0]),label_sym]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.

        # loss, matrix \in R[#data,#classes]
        loss = theano.shared(value=numpy.ones((self.n_data,self.n_classes), 
                        dtype=theano.config.floatX),
                name='cost', borrow=True)
        T.set_subtensor(loss[T.arange(label_sym.shape[0]),label_sym], 0)
        #loss = 0
        # score, matrix \in R[#data,1]
        self.score = T.max(loss + self.compatibility, axis=1)
        margin = T.mean(self.score - self.compatibility[T.arange(label_sym.shape[0]),label_sym])

        return self.l2norm + self.C * margin


    def errors(self, label_sym):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type label_sym: theano.tensor.TensorType
        :param label_sym: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if label_sym has same dimension of y_pred
        if label_sym.ndim != self.y_pred.ndim:
            raise TypeError('label_sym should have the same shape as self.y_pred',
                ('label_sym', label_sym.type, 'y_pred', self.y_pred.type))
        # check if label_sym is of the correct datatype
        if label_sym.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, label_sym))
        else:
            raise NotImplementedError()


    def accuracy(self, label_sym):
        """
        Returns accuracy.
        """
        # check if label_sym has the dimension of y_pred
        if label_sym.ndim != self.y_pred.ndim:
            raise TypeError('label_sym should have the same shape as self.y_pred',
                    ('label_sym', label_sym.type, 'y_pred', self.y_pred.type))
        # check if label_sym is of the correct datatype
        if label_sym.dtype.startswith('int'):
            return T.mean(T.eq(self.y_pred,label_sym))
        else:
            raise NotImplementedError()



def sgd_sje(C, learning_rate, n_epochs, 
        batch_size_train, batch_size_valid, batch_size_test,
        dataset_path_train, dataset_path_valid, dataset_path_test):
    """
    SGD for Structured Joint Embedding.

    :type C: float
    :param C: the data fidelity hyperparameter

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type batch_size_train: int
    :param batch_size_train: the batch size

    :type batch_size_valid: int
    :param batch_size_valid: the batch size

    :type batch_size_test: int
    :param batch_size_test: the batch size

    :type dataset_path_train: string
    :param dataset_path_train: the path of the training dataset
        dataset must be a quintuplet 
        (x,l,label_sym,train_test_label_sym,train_test_l) where
        * x - input embedding
        * l - class 
        * label_sym - output embedding (mapping from class to vector
          attributes)
        * train_test_label_sym - output embedding for train/test
        * train_test_l - classes for train/test

    :type dataset_path_valid: string
    :param dataset_path_valid: the path of the test dataset
        format the same as dataset_train

    :type dataset_path_test: string
    :param dataset_path_test: the path of the test dataset
        format the same as dataset_train
    """

    def compute_number_batches(n_data, batch_size):
        if batch_size is numpy.Inf:
            return 1
        else:
            n_batches = n_data / batch_size
            if n_batches == 0:
                return 1
            else:
                return n_batches


    dataset_train = load_data(dataset_path_train)
    #dataset_valid = load_data(dataset_path_valid)
    dataset_test = load_data(dataset_path_test)

    x_train, label_train, y_train, y_full, label_full_train = dataset_train 
    #x_valid, label_valid, y_valid, y_full = dataset_valid
    x_valid, label_valid, y_valid, y_full, label_full_valid = dataset_train
    x_test, label_test, y_test, y_full, label_full_test = dataset_test

    # compute number of minibatches for training, validation and testing
    n_train_batches = compute_number_batches(
            x_train.get_value(borrow=True).shape[0], batch_size_train)
    n_valid_batches = compute_number_batches(
            x_valid.get_value(borrow=True).shape[0],  batch_size_valid)
    n_test_batches = compute_number_batches(
            x_test.get_value(borrow=True).shape[0], batch_size_test)

    n_input = x_train.get_value(borrow=True).shape[1]
    n_output = y_full.get_value(borrow=True).shape[1]
    #n_classes = y_full.get_value(borrow=True).shape[0]
    #n_classes_train = n_classes
    n_classes_train = y_train.get_value(borrow=True).shape[0]
    n_classes_test = y_test.get_value(borrow=True).shape[0]

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the input features
    y = T.matrix('y')  # the label features
    # the labels are presented as 1D vector of [int]
    label_sym = T.ivector('label_sym')  

    # construct the SJE here
    classifier_train = StructuredJointEmbedding(
            C,
            input_embedding=x, output_embedding=y,
            n_input=n_input, n_output=n_output, 
            batch_size=batch_size_train, n_classes=n_classes_train)

    classifier_test = StructuredJointEmbedding(
            C,
            input_embedding=x, output_embedding=y,
            n_input=n_input, n_output=n_output,
            batch_size=batch_size_test, n_classes=n_classes_test)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier_train.negative_log_likelihood(label_sym)
    
    # compiling a Theano function that computes the mistakes 
    # that are made by the model on a minibatch
    model_valid = theano.function(inputs=[index],
            outputs=classifier_train.errors(label_sym),
            givens={
                x: x_valid[index * batch_size_valid:(index + 1) * batch_size_valid],
                y: y_valid,
                label_sym: label_valid[index * batch_size_valid:(index + 1) * batch_size_valid]})

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier_train.W)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier_train.W, classifier_train.W - learning_rate * g_W)]

    # compiling a Theano function `model_train` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    model_train = theano.function(inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: x_train[index * batch_size_train:(index + 1) * batch_size_train],
                y: y_train,
                label_sym: label_train[index * batch_size_train:(index + 1) * batch_size_train]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    #best_params = None
    best_validation_loss = numpy.inf
    error_test = 1.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = model_train(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [model_valid(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                    (epoch, minibatch_index + 1, n_train_batches,
                    this_validation_loss * 100.))
                print('train cost is: %f' % minibatch_avg_cost)

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss

                    # transfer the weights from the train to the test model
                    classifier_test.W = classifier_train.W
                    classifier_test.define_model()

                    model_test = theano.function(inputs=[index],
                        outputs=classifier_test.errors(label_sym),
                        givens={
                            x: x_test[index * batch_size_test: (index + 1) * batch_size_test],
                            y: y_test,
                            label_sym: label_test[index * batch_size_test: (index + 1) * batch_size_test]})

                    # test the model
                    test_losses = [model_test(i)
                                   for i in xrange(n_test_batches)]
                    error_test = numpy.mean(test_losses)

                    #import ipdb
                    #ipdb.set_trace()

                    print(('     epoch %i, minibatch %i/%i, test error of best'
                       ' model %f %% (accuracy %f %%)') %
                        (epoch, minibatch_index + 1, n_train_batches,
                         error_test * 100., (1.0 - error_test) * 100.))

            if patience <= iter:
                done_looping = True
                break

        
        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs.
        learning_rate = learning_rate * 0.99 
        print 'learning rate: ', learning_rate

        updates = [(classifier_train.W,classifier_train.W - learning_rate*g_W)]
        model_train = theano.function(inputs=[index],
                outputs=cost,
                updates=updates,
                givens={
                    x: x_train[index * batch_size_train:(index + 1) * batch_size_train],
                    y: y_train,
                    label_sym: label_train[index * batch_size_train:(index + 1) * batch_size_train]})

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%, '
           'with test error %f %% (accuracy %f %%)') %
                 (best_validation_loss * 100., error_test * 100., (1.0 - error_test)*100.0))
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))


if __name__ == '__main__':
    # hyperparameters
    C = 10.0
    learning_rate = .1
    n_epochs = 450
    #batch_size_train = 1771 #number that divides #data
    batch_size_train = 8855
    batch_size_valid = batch_size_train
    batch_size_test =  2933
    dataset_path_train = os.path.join('data','train_data.p')
    dataset_path_valid = dataset_path_train
    dataset_path_test = os.path.join('data','test_data.p')

    # SGD
    sgd_sje(
            C=C,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            batch_size_train=batch_size_train,
            batch_size_valid=batch_size_valid,
            batch_size_test=batch_size_test,
            dataset_path_train=dataset_path_train,
            dataset_path_valid=dataset_path_valid,
            dataset_path_test=dataset_path_test)

