import tensorflow as tf
import pandas as pd
import numpy as np

trainx = pd.read_csv("usps_trainx.data", header=None, delimiter=r"\s+")
trainy = pd.read_csv("usps_trainy.data", header=None, delimiter=r"\s+")
testx = pd.read_csv("usps_testx.data", header=None, delimiter=r"\s+")
testy = pd.read_csv("usps_testy.data", header=None, delimiter=r"\s+")

im_height = 16 #image height
im_width = 16 #image width
num_classes = 10 #number of classes for classificaiton

#normalise the images
trainx = trainx / 256
testx  = testx  / 256

#transform pandas to numpy arrays
trainx = trainx.as_matrix()
trainy = trainy.as_matrix()
testx = testx.as_matrix()
testy = testy.as_matrix()

#remove the second dimensions:
trainy = trainy[:,0]
testy = testy[:,0]

#randomly shuffle the train data
import random
random_idx = random.sample([x for x in range(len(trainx))],len(trainx))
trainx = trainx[random_idx,]
trainy = trainy[random_idx]

#Transform the images in a 16x16 form; the resulting tensor would be 2000x16x16 both for training and for test data.
trainx = trainx.reshape(len(trainx),16,16)
testx = testx.reshape(len(trainx),16,16)

#1-hot encode the labels trainy and testy
trainy = pd.get_dummies(trainy).as_matrix()
testy = pd.get_dummies(testy).as_matrix()

#Extract batches funciton that returns a dictionary ready to be fed to the model

def next_batch(data_x,data_y,indices):
    return data_x[indices,:,:] , data_y[indices,:]
    
# Defining the convolution operation (with relu activation), and the maxpool operation
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/convolutional_network.ipynb

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# Construct the computational graph

batch_size = 5
num_epochs = 400
learning_rate = 0.001

conv1_size = 64 #number of outputs from the conv layer
c = 5 #convolution window size
stride = 2 #stride of the convolution

conv2_size = 32 #number of outputs from the 2nd conv layer
c2 = 5 #2nd convolution window size
stride_2 = 2 #stride of the 2nd convolution

full_conn_size = 1024 #number of nodes in the fully connedted layer

#CONSTRUCT THE GRAPH
with tf.Graph().as_default():
    #introduce placeholders for the data(images and labels) to later feed batches into
    X = tf.placeholder(tf.float32, shape=(batch_size,im_height,im_width)) #images
    y = tf.placeholder(tf.int32, shape=(batch_size,num_classes)) #labels
    
    #transform input into a 4d tensor to include the colour channels
    X_prime = tf.reshape(X, shape=[-1, im_height, im_width, 1])
    
    #define the parameters(variables) of the model
    #for the convolution layer:
    W1 = tf.Variable(tf.random_normal([c, c, 1, conv1_size]))
    b1 = tf.Variable(tf.random_normal([conv1_size]))
    #for the second convolution layer:
    W2 = tf.Variable(tf.random_normal([c2, c2, conv1_size, conv2_size]))
    b2 = tf.Variable(tf.random_normal([conv2_size]))
    #for the fully connected layer:
    W3 = tf.Variable(tf.random_normal([1*1*conv2_size, full_conn_size]))
    b3 = tf.Variable(tf.random_normal([full_conn_size]))
    #for the output:
    WO = tf.Variable(tf.random_normal([full_conn_size, num_classes]))
    bO = tf.Variable(tf.random_normal([num_classes]))
    
    #define the model structure
    # c x c convolution, 1 input, conv1_size outputs
    convolution = conv2d(X_prime, W1, b1, stride)
    # then apply pooling
    pool = maxpool2d(convolution)
    
    #one more convolution+pooling layer:
    convolution2 = conv2d(pool, W2, b2, stride_2)
    # then apply pooling
    pool2 = maxpool2d(convolution2)
    
    #add a fully connected layer
    # Reshape convolution output to fit the fully connected layer input
    full_conn = tf.reshape(pool2, [-1, 1*1*conv2_size])
    full_conn = tf.add(tf.matmul(full_conn, W3), b3)
    full_conn = tf.nn.relu(full_conn)
    
    # Output, class prediction
    predictions = tf.add(tf.matmul(full_conn, WO), bO)
    
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Evaluate model
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/convolutional_network.ipynb
    correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # Feed Dict and Run session
    num_batches = int(len(trainx)/batch_size)

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)
    # Feed the batches one by one
    for t in range(num_epochs):
        for i in range(num_batches):
            batch_X, batch_y = next_batch(trainx,trainy,[x for x in range(len(trainx))][i*batch_size:(i+1)*batch_size])
            sess.run(optimizer, feed_dict={X: batch_X, y: batch_y})

#    for _ in range(1000):
#        batch_xs, batch_ys = mnist.train.next_batch(100)
#        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#Evaluating the algorithm. Note: This is not elegant since I couldn't find a way to introduce a tensor of size depending on other tensor in the Graph above

num_test_batches = int(len(testx)/batch_size)
accuracy_ = np.zeros(num_test_batches)

for i in range(num_test_batches):
        batch_X, batch_y = next_batch(testx,testy,[x for x in range(len(testx))][i*batch_size:(i+1)*batch_size])
        accuracy_[i] = sess.run(accuracy, feed_dict={X: batch_X, y: batch_y})
        
print("The test accuracy is ",sum(accuracy_)/num_test_batches)
