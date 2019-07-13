import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/MNIST/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_true = tf.placeholder(tf.float32, [None, 10])

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

logits = tf.matmul(x, w) + b
y = tf.nn.softmax(logits)

xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
loss = tf.reduce_mean(xent)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimize = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 128

def training_step (iterations):
    for i in range (iterations):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        feed_dict_train = {x: x_batch, y_true: y_batch}
        sess.run(optimize, feed_dict=feed_dict_train)

def test_accuracy ():
    feed_dict_test = {x: mnist.test.images, y_true: mnist.test.labels}
    acc = sess.run(accuracy, feed_dict=feed_dict_test)
    print('Testing accuracy:', acc)

training_step(100)
test_accuracy()