'''
           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                   Version 2, December 2004

Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>

Everyone is permitted to copy and distribute verbatim or modified
copies of this license document, and changing it is allowed as long
as the name is changed.

           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
  TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

 0. You just DO WHAT THE FUCK YOU WANT TO.
'''


from datetime import datetime
from csv import DictReader, writer, reader
from math import exp, log, sqrt
import os
import sys
import getopt

# TL; DR, the main training process starts on line: 250,
# you may want to start reading the code from there


##############################################################################
# parameters #################################################################
##############################################################################
opts, args = getopt.getopt(sys.argv[1:], "a:b:l:m:e:t:u:s:h:i:x:", ["alpha=", "beta=", "L1=", "L2=", "epoch=", "train=", "test=", "submission=", "non_feature_cols=", "non_factor_cols=", "text_cols="])
opts = {x[0]:x[1] for x in opts}
#print opts

if bool(opts['--non_feature_cols']):
    non_feature_cols = [x.strip() for x in opts['--non_feature_cols'].split(',')]
else:
    non_feature_cols = []

if bool(opts['--non_factor_cols']):
    non_factor_cols = [x.strip() for x in opts['--non_factor_cols'].split(',')]
else:
    non_factor_cols = []

if bool(opts['--text_cols']):
    text_cols = [x.strip() for x in opts['--text_cols'].split(',')]
else:
    text_cols = []

# A, paths
train = opts['--train']
test = opts['--test']
submission = opts['--submission']  # path of to be outputted submission file

# B, model
alpha = float(opts['--alpha'])  # learning rate
beta = float(opts['--beta'])    # smoothing parameter for adaptive learning rate
L1 = float(opts['--L1'])        # L1 regularization, larger value means more regularized
L2 = float(opts['--L2'])        # L2 regularization, larger value means more regularized

# C, feature/hash trick
D = 2 ** 24             # number of weights to use
interaction = False     # whether to enable poly2 feature interactions

# D, training/validation
epoch = int(opts['--epoch'])           # learn training data for N passes

##############################################################################
# class, function, generator definitions #####################################
##############################################################################

class ftrl_proximal(object):
    ''' Our main algorithm: Follow the regularized leader - proximal

        In short,
        this is an adaptive-learning-rate sparse logistic-regression with
        efficient L1-L2-regularization

        Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    '''

    def __init__(self, alpha, beta, L1, L2, D, interaction):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D
        self.interaction = interaction

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.] * D
        self.z = [0.] * D
        self.w = {}

    def _indices(self, x):
        ''' A helper generator that yields the indices in x

            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        '''

        # first yield index of the bias term
        yield 0

        # then yield the normal indices
        for index in x:
            yield index

        # now yield interactions (if applicable)
        if self.interaction:
            D = self.D
            L = len(x)

            x = sorted(x)
            for i in xrange(L-1):
                for j in xrange(i+1, L):
                    # one-hot encode interactions with hash trick
                    yield abs(hash(str(x[i]) + '_' + str(x[j]))) % D

    def predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: a list of feature indices

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''

        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = {}

        # wTx is the inner product of w and x
        wTx = 0.
        j = 0
        for i in self._indices(x):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]
            j += 1

        # cache the current w for update stage
        self.w = w

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        ''' Update model using x, p, y

            INPUT:
                x: a list of feature indices
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''

        # parameter
        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        w = self.w

        # gradient under logloss
        #g = -y*(1 - p)
        if y == 1.0:
            g = 1.0*(p - y)
        else:
            g = 1.0*(p - y)

        # update z and n
        for i in self._indices(x):
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g

def logloss(p, y):
    ''' FUNCTION: Bounded logloss

        INPUT:
            p: our prediction
            y: real answer

        OUTPUT:
            logarithmic loss of p given y
    '''

    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)


def data(path, D):
    ''' GENERATOR: Apply hash-trick to the original csv row
                   and for simplicity, we one-hot-encode everything

        INPUT:
            path: path to training or testing file
            D: the max index that we can hash to

        YIELDS:
            ID: id of the instance, mainly useless
            x: a list of hashed and one-hot-encoded 'indices'
               we only need the index since all values are either 0 or 1
            y: target
    '''
    for t, row in enumerate(DictReader(open(path))):
        # process patient_id
        if 'ID' in row:
            ID = row['ID']
            del row['ID']
        else:
            ID = 0

        # process target
        y = 0.
        if 'TARGET' in row:
            if row['TARGET'] == '1':
                y = 1.
            del row['TARGET']

        for col in non_feature_cols:
            if col in row:
                del row[col]
 
        # build x
        x = []
        for key in row:
            if key in non_factor_cols:
                try:
                    value = float(row[key])
                except TypeError:
                    value = 0.0
                index = abs(hash(key + '_12345')) % D
                x.append(index)
            elif key in text_cols:
                for word in row[key].split():
                    value = str(word)
                    # one-hot encode everything with hash trick
                    index = abs(hash(key + '_' + value)) % D
                    x.append(index)
            else:
                value = str(row[key])
                # one-hot encode everything with hash trick
                index = abs(hash(key + '_' + value)) % D
                x.append(index)
        #index = abs(hash(str(row['device_conn_type']) + '_int1_' + str(row['app_id']))) % D
        #x.append(index)

        yield t, ID, x, y


##############################################################################
# start training #############################################################
##############################################################################

start = datetime.now()

# initialize ourselves a learner
learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction)
# start training
for e in xrange(epoch):
    loss = 0.
    localcount = 0
    for t, ID, x, y in data(train, D):  # data is a generator
    #    t: just a instance counter
    #    ID: id provided in original data
    #    x: features
    #    y: target

        # step 1, get prediction from learner
        p = learner.predict(x)

        #print progress
        localcount += 1
        if localcount % 100000 == 0:
            print "train: " + str(localcount)

        # step 2, update weights
        learner.update(x, p, y)
        
    #print('Epoch %d finished, elapsed time: %s' % (
    #    e, str(datetime.now() - start)))


##############################################################################
# start testing ##############################################################
##############################################################################
with open(submission, 'w') as outfile:
    outfile.write('ID,PRED\n')
    for t, ID, x, y in data(test, D):
        p = learner.predict(x)
        outfile.write('%s, %f\n' % (str(ID), p))

