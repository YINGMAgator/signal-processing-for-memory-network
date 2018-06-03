#************************* data process begin******************************/
import os
import re
import pickle
import numpy as np
import pdb
import tensorflow as tf
# rnn model
batchSize = 1
lstmUnits = 2
# numClasses = 2
iterations = 20000
maxSeqLength = 100
numDimensions = 1 #Dimensions for each word vector
hiddendimension=1
tf.reset_default_graph()
label= tf.placeholder(tf.float32, [batchSize, maxSeqLength,numDimensions])
data = tf.placeholder(tf.float32, [batchSize, maxSeqLength,numDimensions])
inputmask = tf.placeholder(tf.int32, [batchSize, maxSeqLength,1])

# lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
# lstmCell = tf.contrib.rnn.BasicRNNCell(lstmUnits)
# value, value1 = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
#

# weighti = tf.Variable(tf.constant(0.1, shape=[hiddendimension]))

# value=tf.zeros([batchSize, maxSeqLength,lstmUnits])
weighti = tf.Variable(tf.constant(0.1, shape=[1,lstmUnits]))
weightj=np.array([[0.1,0.1]])
weighth = tf.Variable(tf.truncated_normal([lstmUnits, 2]))
biash = tf.Variable(tf.constant(0.1, shape=[1,2]))

weight = tf.Variable(tf.truncated_normal([lstmUnits, 1]))
bias = tf.Variable(tf.constant(0.1, shape=[hiddendimension]))
# bias=0.1
# weight=np.array([[-0.03735173], [ 1.30852091]])

hh=[tf.zeros([lstmUnits])]
output_list = []
for position in range(maxSeqLength):

    # h= tf.matmul(h, weighth) + biash
    hh=tf.matmul([data[0][position]], weighti) + tf.matmul(hh, weighth) + biash
    output_list.append(hh[0])
outputs = tf.stack(output_list)
output_value=tf.expand_dims(outputs, 0)
    # h=value[0,position,0:1]
l_out_x = tf.reshape(output_value, [-1, lstmUnits], name='2_2D')
# shape = (batch * steps, output_size)
pred = tf.nn.tanh(tf.matmul(l_out_x, weight) + bias)
#layer2
# weight1 = tf.Variable(tf.truncated_normal([hiddendimension, numDimensions]))
# bias1 = tf.Variable(tf.constant(0.1, shape=[numDimensions]))
# pred = tf.matmul(pred1, weight1) + bias1


labels=tf.reshape(label, [-1, numDimensions])
mask=tf.reshape(inputmask, [-1, 1])
loss=tf.losses.mean_squared_error(
    labels,
    pred,
    weights=mask,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
)
optimizer = tf.train.AdamOptimizer().minimize(loss)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# import datetime
# merged = tf.summary.scalar('Losstrain', loss)
# logdir = "tensorboard_train/" + "rnn/"
# writer = tf.summary.FileWriter(logdir, sess.graph)

Fs = 32
# inputdata = np.zeros([batchSize, maxSeqLength,1])
# inputlabel = np.zeros([batchSize, maxSeqLength,1])
mask = np.zeros([batchSize, maxSeqLength,1])
# x = np.arange(maxSeqLength)
# y = np.sin(2 * np.pi * x /2/Fs)
# y1 = np.sin(2 * np.pi* x /Fs)
# y.resize(maxSeqLength,1)
# y1.resize(maxSeqLength,1)
# # inputdata[0][0:maxSeqLength] = y
# inputlabel[0][0:maxSeqLength] = y1
x = np.arange(maxSeqLength)
from scipy import signal
# def case1():
inputdata = np.zeros([batchSize, maxSeqLength,1])
inputlabel = np.zeros([batchSize, maxSeqLength,1])
x = np.arange(maxSeqLength)
y = np.sin(2 * np.pi * x /Fs)
y1 = np.sin(2 * np.pi* x /2/Fs)

# y = signal.sawtooth(2 * np.pi * x /Fs)
# y1 = signal.sawtooth(2 * np.pi* x /2/Fs)
y.resize(maxSeqLength,1)
y1.resize(maxSeqLength,1)
# inputdata[0][0:maxSeqLength] = (y)
# inputdata[0][16:32] = y[0:16]
# inputdata[0][17:32] = -y[17:32]*2
# inputlabel[0][0:maxSeqLength] = abs(y1)
inputlabel[0][0:maxSeqLength] = (y1)
n=48
mask[0][0:48] =[[1]] * 48

weighitrackh1 = []
weighitrackh2 = []
weighitrackh3 = []
weighitrackh4 =[]
weighitrackh5 =[]
weighitrackh6 =[]
biastrackh1 = []
biastrackh2 = []

weighitracko1 = []
weighitracko2 = []
biastracko =[]
ls=1
acc=0.1
import matplotlib.pyplot as plt
for i in range(iterations):
    # case=np.random.randint(1)
    # x,y,y1,inputdata,inputlabel=switch[case]
    # pdb.set_trace()
    sess.run(optimizer, {data: inputdata, label: inputlabel, inputmask:mask})

   # print(i)
    # aaa,bbb,ccc=sess.run([label,data,inputmask], {data: arr, label: labels, inputmask:mask})
    if (i % 100 == 0):
        if (ls<acc):
            n=n+16
            acc=acc/2
            mask[0][0:n] =[[1]] * n
            print ('increasing training length',n)

            pdb.set_trace()
        # x,y,y1,inputdata,inputlabel=switch[case]
        variables_names =[v.name for v in tf.trainable_variables()]
        values = sess.run(variables_names)
        # for k,v in zip(variables_names, values):
        #     print(k, v)
        weighitrackh1.append(values[0][0][0])
        weighitrackh2.append(values[0][0][1])
        weighitrackh3.append(values[1][0][0])
        weighitrackh4.append(values[1][0][1])
        weighitrackh5.append(values[1][1][0])
        weighitrackh6.append(values[1][1][1])
        biastrackh1.append(values[2][0][0])
        biastrackh2.append(values[2][0][1])

        weighitracko1.append(values[3][0][0])
        weighitracko2.append(values[3][1][0])
        biastracko.append(values[4][0])
        ls,pd,v,v1,weighti_d = sess.run([loss,pred,output_value,l_out_x,weighti], {data: inputdata, label: inputlabel, inputmask:mask})
        print (ls)


        plt.close()
        plt.ion()
        plt.plot(x,inputlabel[0])
        plt.plot(x,pd,linestyle='-.')
        plt.legend(['sin(t/2)','output'])
        plt.ylim(ymax=1.4)
        plt.draw()
        plt.savefig('rnnhalf')
        # #
        plt.close()
        plt.ion()
        plt.subplot(4,1,1)
        plt.plot(weighitrackh5)
        plt.plot(weighitrackh6)
        plt.legend(['wh5','wh6'])

        plt.subplot(4,1,2)
        plt.plot(weighitrackh1)
        plt.plot(weighitrackh2)
        plt.plot(weighitrackh3)
        plt.plot(weighitrackh4)
        plt.legend(['wh1','wh2','wh3','wh4'])

        plt.subplot(4,1,3)
        plt.plot(biastrackh1)
        plt.plot(biastrackh2)
        plt.plot(biastracko,linestyle='-.')
        plt.legend(['bh1','bh2','bo'])

        plt.subplot(4,1,4)
        plt.plot(weighitracko1)
        plt.plot(weighitracko2)
        plt.legend(['wo1','wo2'])

        plt.draw()
        plt.savefig('weightracking')
