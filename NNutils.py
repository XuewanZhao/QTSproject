# -*- coding: utf-8 -*-
"""
Created on Wed May  9 09:19:01 2018

@author: Xuewan Zhao
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 18:22:35 2018

@author: Xuewan
"""

'''
features:
    open, high, low,close,MA,RSI,fastRSI,slowRSI,wR.

RSI = 100 - 100 / (1 + RS)
Where RS = Average gain of up periods during the specified time frame / Average loss of down periods during the specified time frame"
StochRSI = (RSI - Lowest Low RSI) / (Highest High RSI - Lowest Low RSI)
%R = (Highest High - Close)/(Highest High - Lowest Low) * -100

reference:https://www.slideshare.net/t_koshikawa/predition
https://piazza-resources.s3.amazonaws.com/jct4erta5sg67d/jg82wx6wdyu309/HW9.html?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAJMJFCQJYWJS7AWVQ%2F20180430%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20180430T214300Z&X-Amz-Expires=10800&X-Amz-SignedHeaders=host&X-Amz-Security-Token=FQoDYXdzEPb%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDPl1gs8okSqI%2BmkQRCK3A1O16AxWKR9BC%2BjlHW94dTTd%2BM3le%2FlFZfJNmA8FP34TRynzZWk%2Fe2VqHHAcYTfjfpEkx7C0mFFh2OrnLejnLkz7NhFkc9TnPped8Y%2FKDU2vnlFaMTzgCycclZbZbVtax3xB%2Biki5Wht6utl6T%2FnnB5C3SPuv%2Fz7xI8gR0sPFj1i5IVT33O5ak4GtQCWF5o%2B2A8u4tzHTmQ4RzvG2ka%2FTBbVq2HpT3ad3WOxikxc%2B2mNNOu6ZreRALWMPZLzRJ77G21Ysb8Z165eLTXpblyZ8lBUaq91mR%2BHYnhFFd5%2FVcKcjcy%2FbndSXEjPryCiE7zsqSr8%2BFS%2B7pqbOns46%2BpCFfMv2LeMvWsW3Ck1qnfal%2BF2791eBTlIaYgjRFaYeIqthMBiymipxzmLjQ1mhSZQYLRSY1a68L0xoZ1rj%2B07TJT4AC2z5qJwiVpT6OInPLDLIpPl7QnBwgmMmDasTgoZAfGwcuDn%2BFE51ehqx1JNXnEsKUN2Q2PaqPS2IH%2BvFreYtonk2ruIod5PZFC6jNgDLuEj%2Fpu1dKKoGw6mtUfUHCmrJEs8j0qkFWcXQQXEBA7TZJxTu7r2fDcowYye1wU%3D&X-Amz-Signature=3aad528cdcaf8ae09b3f450c265db95bc26c43838464765f1ca00148d7af66f6
'''

import pandas as pd
import numpy as np

# Calculate MA
def MA(df,period):
    df['MA'] = df.Close.rolling(period).mean()
    return df

# Calculate RSI
def RSI(df,period):
    dUP,dDown = (df.Close - df.Close.shift(1)).copy(),(df.Close - df.Close.shift(1)).copy()
    dUP[dUP<0] = 0
    dDown[dDown>0] = 0
    
    RolUP = dUP.rolling(period).mean()
    RolDown = dDown.rolling(period).mean().abs()
    
    RS = RolUP/RolDown
    
    df['RSI'] = 100 - 100/(1+RS)
    return df

# Calculate stochastic RSI
def StocRSI(df,period,type_ = "fast"):
    # type_ is for the period of the stochastic RSI: fast, slow
    # fast should be smaller than slow. These two should all be calculated.
    LowRSI = df.RSI.rolling(period).min()
    HighRSI = df.RSI.rolling(period).max()
    df[type_ + 'RSI'] = (df.RSI - LowRSI)/(HighRSI - LowRSI)
    return df

# Calculates william %R.
def williamR(df,period):
    High = df.Close.rolling(period).max()
    Low = df.Close.rolling(period).min()
    df['wR'] = (High - df.Close)/(High - Low)*(-100)
    return df

# Calculate class.
def trend(df):
    diffMA = pd.DataFrame()
    diffMA['dif'] = df.MA.shift(-1) - df.MA
    diffMA['class_'] = 0
    diffMA.loc[diffMA.dif>0,'class_'] = 1
    df['class_'] = diffMA.class_
    return df

def GenData(df,MAperiod,fRperiod,sRperiod):      
    '''
    parameters:
    filename: data file
    MAperiod: moving average period
    fRperiod: fast RSI period,which should be smaller than short RSI period
    sRperiod: short RSI period
    
    By now, we have a dataframe contains all features we need:
        Open, High, Low, Close, MA, RSI, fastRSI, shortRSI, wR
        feature 'class_' means the MA go up or down(1,0) next time. It's used for training and score our model.
    '''
    df = MA(df,MAperiod)
    df = RSI(df,MAperiod)
    df = StocRSI(df,fRperiod,'fast')
    df = StocRSI(df,sRperiod, 'short')
    df = williamR(df,MAperiod)
    df = trend(df)
    return df
    
################
# Train NN
################
    
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import tensorflow as tf 

def RollingTrain(df,lookback_period,holding_period):
    x = np.array(df.drop(['class_','wR'],axis = 1).dropna())
    y = np.array(df.class_[len(df) - len(x):])
    df['pred'] = 0
    df['position'] = 0
    for i in range(lookback_period,len(x)-1,holding_period):
        x_train = x[i - lookback_period:i]
        y_train = y[i - lookback_period:i]
        x_test = x[i:i+5]
        y_test = y[i:i+5]
    #x_train, x_test, y_train, y_test = train_test_split(x,y, test_size)
        stats,pred = TrainNN(x_train,y_train,x_test,y_test)
        df.loc[i,'pred'] = np.mean(pred)
        if np.mean(pred)>0.5:
            df.loc[i,'pred'] = 1
            df.loc[i：i+holding_period,'position'] = 1
        else:
            df.loc[i,'pred'] = -1
            df.loc[i：i+holding_period,'position'] = -1
    return df


def TrainNN(x_train,y_train,x_test,y_test):
    '''
    clf = MLPClassifier(activation = 'tanh',solver = 'lbfgs',hidden_layer_sizes = (4,),random_state = 0)
    clf.fit(x_train,y_train)
    # We try to plot the configuration by function from Git.
    from draw_neural_network import draw_neural_net
    fig = plt.figure(figsize = (12,12))
    ax = fig.gca()
    ax.axis('off')
    layer_sizes = [2,4,1]
    draw_neural_net(ax,.1,.9,.1,.9,layer_sizes,clf.coefs_,clf.intercepts_,clf.n_iter_,clf.loss_)
    fig.savefig("nn_diagram.png")
    
    # Step3: Predict and get the score.
    clf.predict(x_test)
    print(clf.score(x_test,y_test))
    '''
    tf.reset_default_graph()
    x = tf.placeholder('float32', shape=(None,x_train.shape[1]), name='x') 
    y = tf.placeholder('int64', (None, ), name='y') 
    is_training = tf.placeholder(tf.bool, shape=(), name='is_training') 
    print('Shape of "x" =',x.shape) 
    print('Shape of "y" =',y.shape) 
    print('Shape of "is_training" =',is_training.shape) 
    
    #   2) Creat layers.
        # scale inputs 
    # scale inputs
    scaled = tf.contrib.layers.batch_norm(x, center=True,
                                          scale=True, scope='scaled',
                                          is_training=is_training)
    # hidden layer 1
    hidden1 = tf.contrib.layers.fully_connected(scaled,
                                                num_outputs=5, scope='hidden1',
                                                activation_fn=tf.nn.relu)
    # hidden layer 2
    hidden2 = tf.contrib.layers.fully_connected(hidden1,
                                                num_outputs=10, scope='hidden2',
                                                activation_fn=tf.nn.relu)
    
    logits = tf.contrib.layers.fully_connected(hidden2,
                                               num_outputs=2, scope='logits',
                                               activation_fn=None)
    output = tf.nn.softmax(logits, name='output')
    
    logloss = tf.losses.log_loss(labels=y,
                                 predictions=output[:,1],
                                 weights=1.0,
                                 epsilon=1e-07,
                                 scope=None,
                                 loss_collection=tf.GraphKeys.LOSSES)
    true_pred = tf.equal(tf.argmax(output,1), y)
    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(true_pred, "float"))
    
    n_epochs = 1000
    best_epoch = 0
    batch_size = 25
    best_test_logloss = float("inf")
    test_accuracy_for_best_test_logloss = 0
    history = []
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
    with tf.control_dependencies(update_ops):
        # Ensures that we execute the update_ops before performing the train_step
        optimizer = tf.train.AdamOptimizer()
        # Select type of optimizer
        train_step = optimizer.minimize(logloss)
        # Gradient descent step
    sess = tf.Session()
    sess.run(tf.global_variables_initializer()) # Initialize all variables
    for epoch in range(n_epochs):
        batch_from = 0
        if epoch % 100 == 0:
            print ("Current epoch %i " % epoch)
        while batch_from < len(y_train):
            batch_to = min(batch_from + batch_size, len(y_train))
            x_trainbatch = x_train[batch_from:batch_to,]
            y_trainbatch = y_train[batch_from:batch_to]
            batch_from = batch_from + batch_size # for next iteration
            sess.run(train_step,
                    feed_dict={x: x_trainbatch,
                               y: y_trainbatch,
                               is_training: True})
        # Run batches through the network until epoch is over
        # Calculate statistics (logloss & accuracy) for train & test:
        train_res = sess.run([logloss, accuracy],
                              feed_dict={x: x_train, y: y_train, 
                                         is_training: False})
        test_res = sess.run([logloss, accuracy],
                             feed_dict={x: x_test, 
                                        y: y_test, is_training: False})
        history += [[epoch] + train_res[:] + test_res[:]]
        # Update history and write to the log
        test_logloss = test_res[0]
        test_accuracy = test_res[1]
        if test_logloss < best_test_logloss:
        # save params for best test logloss only
            best_epoch = epoch
            best_test_logloss = test_logloss
            test_accuracy_for_best_test_logloss = test_accuracy
    print('Best epoch is #{} with test logloss of {:.4f} test_accuracy = {:.4f})'\
          .format(best_epoch, best_test_logloss,
                  test_accuracy_for_best_test_logloss))
    
    stats=pd.DataFrame(history)
    pred = output.eval(feed_dict={x: x_test, is_training:False},session=sess)
    '''
    # blue for class 0, orange for class 1
    color = ['blue' if y == 0 else 'orange' for y in y_test]
    plt.scatter(range(y_test.shape[0]), pred[:,1], color=color)
    plt.title('Probability of class 1')
    plt.show()
    '''
    return stats,pred[:,1]
    
def DrawStats(stats):
    # Plot logloss.
    plt.plot(stats.iloc[:,0], stats.iloc[:,1], label='train')
    plt.plot(stats.iloc[:,0], stats.iloc[:,3], label='test')
    plt.title('logloss')
    plt.legend()
    plt.show()
    
    # Plot accuracy for train & test
    plt.plot(stats.iloc[:,0], stats.iloc[:,2], label='train')
    plt.plot(stats.iloc[:,0], stats.iloc[:,4], label='test')
    plt.title('accuracy')
    plt.legend()
    plt.show()
    return None

