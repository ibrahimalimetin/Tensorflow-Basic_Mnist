#ibrahim Ali Metin
import tensorflow as tf #kullanacağımız kütüphaneyi import ettik (tensorflow google 'ın makine öğrenimi için geliştirdiği açık kaynak kodlu bir kütüphanedir.)

from tensorflow.examples.tutorials.mnist import input_data #üzerinde çalışacağımız verisetini ekliyoruz. mnist içerisinde 70000 resim var 28*28 boyutlu
mnist = input_data.read_data_sets("data/MNIST/", one_hot=True)#veri setini aldık. one_hot elemanları alırken 10 elemanlı olarak almaktadır. (8=0000000010)

x = tf.placeholder(tf.float32, [None,784]) #gelen resimleri atacağımız yer tutucular
y_true = tf.placeholder(tf.float32, [None,10]) #resimler 10 uzunluğunda
#sadece input ve inputun labellerini alma işlemini yukarıda yaptık

w = tf.Variable(tf.zeros([784,10]))    #eğitilecek parametleri tanımladık (W)
b = tf.Variable(tf.zeros([10]))        #bias

logits = tf.matmul (x,w) + b #x input, w ve b optimize olacak parametrelerdir.
#softmax aktivasyon fonk. geçireceğiz ve sinir ağımız oluşmuş oluyor:
y = tf.nn.softmax(logits) #nörondaki değerleri sıkıştırarak  0 ile 1 arasında alıyoruz

xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)  #loss fonk ile modelimizin ne kadar doğru ne kadar yanlış olduğunu hesaplıyoruz.
loss = tf.reduce_mean(xent)#ortalama alınır ve loss 'a atılır

correct_pre = tf.equal(tf.arg_max(y,1), tf.arg_max(y_true,1))
accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32)) #tahmin başarım oranı

optimize = tf.train.GradientDescentOptimizer(0.5).minimize(loss) #adım boyutu (başarı oranına etkisi değiştirilebilir)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 128 # her yığında 128 resim alacağız

def training_step(iterations):
    for i in range(iterations):#0 dan başlayarak iterasyon sayısına kadar dönerek eğitim yapacak
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        feed_dict_train = {x : x_batch , y_true : y_batch}
        sess.run(optimize, feed_dict= feed_dict_train)
        #batch size kadar mnist resim ve etiketlerini aldık aldığımız resimleri playsolder lara atayarak modeli optimize ettik ve iterasyon kdar döndü
        #ve eğitim gerçekleştirdik. gözetimli öğrenme yaptığımız için tüm verilerin gerçek değerleri belli idi. sırada test işlemi var:

    def test_accuracy():
        feed_dict_test = {x: mnist.test.images, y_true: mnist.test.labels}
        acc = sess.run(accuracy, feed_dict=feed_dict_test) #test setindeki veriler alınarak eğitilmiş modelden geçirildi.
        print('Test Doğruluk Oranı:', acc)

    training_step(0)
    test_accuracy()




