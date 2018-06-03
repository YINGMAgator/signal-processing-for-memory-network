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
maxSeqLength = 400
numDimensions = 1 #Dimensions for each word vector
hiddendimension=1
tf.reset_default_graph()
label= tf.placeholder(tf.float32, [batchSize, maxSeqLength,numDimensions])
data = tf.placeholder(tf.float32, [batchSize, maxSeqLength,numDimensions])
inputmask = tf.placeholder(tf.int32, [batchSize, maxSeqLength,1])

# lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.BasicRNNCell(lstmUnits)
value, value1 = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
#
weight = tf.Variable(tf.truncated_normal([lstmUnits, hiddendimension]))
bias = tf.Variable(tf.constant(0.1, shape=[hiddendimension]))
# value = tf.transpose(value, [1, 0, 2])
l_out_x = tf.reshape(value, [-1, lstmUnits], name='2_2D')
# shape = (batch * steps, output_size)
pred = tf.matmul(l_out_x, weight) + bias
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
import matplotlib.pyplot as plt

for i in range(iterations):
    # case=np.random.randint(1)
    # x,y,y1,inputdata,inputlabel=switch[case]
    # pdb.set_trace()
    sess.run(optimizer, {data: inputdata, label: inputlabel, inputmask:mask})

   # print(i)
    # aaa,bbb,ccc=sess.run([label,data,inputmask], {data: arr, label: labels, inputmask:mask})
    if (i % 100 == 0):
        if (i%2000==0):
            mask[0][0:n] =[[1]] * n
            print ('increasing training length',n)
            n=n+16
            pdb.set_trace()
        # x,y,y1,inputdata,inputlabel=switch[case]
        ls,pd,v,v1,weight_d,bias_d = sess.run([loss,pred,value,value1,weight,bias], {data: inputdata, label: inputlabel, inputmask:mask})
        # pdb.set_trace()
      # writer_train.add_summary(summary, i)
      # # pdb.set_trace()
      # nextBatch, nextBatchLabels,nextmask = getTestBatch();
      # accuracy_test,summary_test = sess.run([accuracy,merged_test], {input_data: nextBatch, input_labels: nextBatchLabels, inputmask:nextmask})
      # writer_test.add_summary(summary_test, i)
        # print(v[0,:,0])
        # print(weight_d)
        # print(bias_d)
        print (ls)


        plt.close()
        plt.ion()
        plt.plot(x,inputlabel[0])
        plt.plot(x,pd,linestyle='-.')
        plt.legend(['sin(t/2)','output'])
        plt.ylim(ymax=1.4)
        plt.draw()
        plt.savefig('rnnhalf')


        f = open('rnnfrequencychange.pckl', 'wb')
        pickle.dump([x,y,y1,v,pd,inputdata,inputlabel], f)
        f.close()
#
#
# plt.close()
# plt.ion()
# plt.plot(x,y1)
# plt.plot(x,pd,linestyle='-.')
# plt.legend(['sin2t','predict'])
# plt.ylim(ymax=1.4)
# plt.draw()
# pdb.set_trace()
# plt.savefig('frequencydouble')
#
#
# plt.close()
# plt.ion()
# plt.plot(x,y1)
# plt.plot(x,y,x, v[0,:,0],x,v[0,:,1])
# plt.legend(['sint','h1','h2'])
# plt.ylim(ymax=1.4)
# plt.draw()
# pdb.set_trace()
# plt.savefig('frequencydoubleh')

# writer.close()

#************************* training process end******************************/
