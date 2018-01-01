'''
文件二：神经网络的训练程序
'''
import os
import tensorflow as tf
import mnist_inference
from tensorflow.examples.tutorials.mnist import input_data

#配置神经网络参数
REGULARAZTION_RATE = 0.0001    #描述模型复杂度的正则化项在损失函数中的系数
MOVING_AVERAGE_DECAY = 0.99
LEARNING_RATE_BASE = 0.8    #基础的学习率
BATCH_SIZE = 100
LEARING_RATE_DECAY = 0.99    #学习率的衰减率
TRAINING_STEP = 10000
MODEL_SAVE_PATH = 'D:\ML\mnist_samp\model'
MODEL_NAME = 'model.ckpt'

def train(mnist):
    #定义输入输出placeholder
    x = tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],name='x-input')
    y_ = tf.placeholder(tf.float32,[None,mnist_inference.OUTPUT_NDOE],name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = mnist_inference.inference(x,regularizer)

    #定义存储训练轮数的变量，可以加快训练早期变量的更新速度
    global_step = tf.Variable(0,trainable=False)

    #滑动平均操作数
    #定义一个滑动平均的类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    #在所有没有指定trainabel=False的神经网络的变量上使用滑动平均
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    #当前batch中所有样例的交叉熵平均值（预测值和真实值差距）
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #总损失=交叉熵损失+正则化损失
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    #指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE, LEARING_RATE_DECAY
    )   #global_step当前迭代的轮数
    #优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    #训练模型时，同时更新神经网络的参数和参数的滑动平均值
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')

    #初始化tensorflow持久类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEP):
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            if i % 1000 == 0:
                print('after %d training step,loss on training batch is %g' %(step,loss_value))

                #保存当前的模型
                #global_step参数可以让保存模型的文件名末尾加上训练的轮数
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets('D:\ML\mnist\MNIST_data',one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()

