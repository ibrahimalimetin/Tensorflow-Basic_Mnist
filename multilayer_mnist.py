import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/MNIST/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_true = tf.placeholder(tf.float32, [None, 10])

layer_1 = 128
layer_2 = 64
layer_3 = 32
layer_out = 10

weight_1 = tf.Variable(tf.truncated_normal([784, layer_1], stddev=0.1))
bias_1 = tf.Variable(tf.constant(0.1, shape=[layer_1]))
weight_2 = tf.Variable(tf.truncated_normal([layer_1, layer_2], stddev=0.1))
bias_2 = tf.Variable(tf.constant(0.1, shape=[layer_2]))
weight_3 = tf.Variable(tf.truncated_normal([layer_2, layer_3], stddev=0.1))
bias_3 = tf.Variable(tf.constant(0.1, shape=[layer_3]))
weight_4 = tf.Variable(tf.truncated_normal([layer_3, layer_out], stddev=0.1))
bias_4 = tf.Variable(tf.constant(0.1, shape=[layer_out]))

y1 = tf.nn.relu(tf.matmul(x, weight_1) + bias_1)
y2 = tf.nn.relu(tf.matmul(y1, weight_2) + bias_2)
y3 = tf.nn.relu(tf.matmul(y2, weight_3) + bias_3)
logits = tf.matmul(y3, weight_4) + bias_4
y4 = tf.nn.softmax(logits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
loss = tf.reduce_mean(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y4, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimize = tf.train.AdamOptimizer(0.001).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 128
loss_graph = []

def training_step (iterations):
    for i in range (iterations):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        feed_dict_train = {x: x_batch, y_true: y_batch}
        [_, train_loss] = sess.run([optimize, loss], feed_dict=feed_dict_train)

        loss_graph.append(train_loss)

        if i % 1000 == 0:
            train_acc = sess.run(accuracy, feed_dict=feed_dict_train)
            print('Iteration:', i, 'Training accuracy:', train_acc, 'Training loss:', train_loss)

def test_accuracy ():
    feed_dict_test = {x: mnist.test.images, y_true: mnist.test.labels}
    acc = sess.run(accuracy, feed_dict=feed_dict_test)
    print('Testing accuracy:', acc)

training_step(10000)
test_accuracy()

plt.plot(loss_graph, 'k-')
plt.title('Loss Grafiği')
plt.xlabel('Iterasyon')
plt.ylabel('Loss')
plt.show()