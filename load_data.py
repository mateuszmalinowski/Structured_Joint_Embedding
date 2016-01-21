#!/usr/bin/env python

"""
"""

import cPickle, numpy, theano, theano.tensor as T


def load_data(dataset):
    ''' 
    Loads the dataset.

    In:
        :type dataset: string
        :param dataset: the path to the dataset 

    Out:
        :type (a,b,c,d,e) where 
            a \in R[#data,#visual_features]
            b \in R[#data] 
            c \in R[#classes,#attributes]
            d \in R[#train_test_classes,#attributes]
            e \in R[#classes]
        :param (input embedding, label, output_embedding,
            train_test_output_embeddings, train_test_labels)
    '''

    #############
    # LOAD DATA #
    #############

    print '... loading data'

    # Load the dataset
    with open(dataset,'rb') as f:
        data = cPickle.load(f)

    def shared_dataset(data_xlyz, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        input_embedding,label,output_embedding,full_output_embedding,full_label \
                = data_xlyz
        shared_x = theano.shared(
                numpy.asarray(input_embedding, dtype=theano.config.floatX), 
                borrow=borrow)
        shared_y = theano.shared(
                numpy.asarray(output_embedding, dtype=theano.config.floatX),
                borrow=borrow)
        shared_z = theano.shared(
                numpy.asarray(full_output_embedding, dtype=theano.config.floatX),
                borrow=borrow)
                                                
        shared_label = theano.shared(
                numpy.asarray(label, dtype=theano.config.floatX),
                borrow=borrow)
        shared_full_label = theano.shared(
                numpy.asarray(full_label, dtype=theano.config.floatX),
                borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_label, 'int32'), shared_y, \
                shared_z, T.cast(shared_full_label,'int32')

    return shared_dataset(data)

