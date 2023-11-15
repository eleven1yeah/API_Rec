#循环神经网络RNN中GRU(Gate Recurrent Unit)
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.python.keras import layers
from keras.models import load_model
from keras.utils.vis_utils import plot_model

#设置随机种子
tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

#设置参数
batchsz = 128
total_words = 10000 #所有单词类型
max_review_len = 80 #句子最大长度
embedding_len = 100 #最常出现的单词 独热编码长度

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)

db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)

#print("x_text",x_test)
de_aa=tf.data.Dataset.from_tensor_slices(x_test)
#print("de_aa",de_aa)
de_bb=de_aa.batch(batchsz, drop_remainder=True)
#print("de_bb",de_bb)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batchsz, drop_remainder=True)

#打印数据的形状和标签的最大最小值以及测试集的形状
print('x_train shape:', x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
print('x_test shape:', x_test.shape)


class MyRNN(keras.Model):

    def __init__(self, units):
        super(MyRNN, self).__init__()

        # 这里与layers类型不同，需要对state0和state1两个cell状态分别定义，因为GRU的循环过程
        # 就是从state0进行门运算后产生输出至state1然后更新state0，然后有经过state1返回来，注意这里因为比LSTM少一个门
        # 所以只有一个tf.zeros
        #[b, 128] c和 h的参数
        self.state0 = [tf.zeros([batchsz, units])] #层1 记忆参数
        self.state1 = [tf.zeros([batchsz, units])] #层2 记忆参数

        # embedding层 [b, 80] ==> [b, 80, 100]
        self.embedding = layers.Embedding(total_words, embedding_len,
                                          input_length=max_review_len)

        # 这里因为是以cell形式构建的，所以需要不能用sequential形式构建网络，这里逐个定义cell0和cell1
        #简单循环神经网络单元 [b, 80, 100] ==> h_dim:128
        self.rnn_cell0 = layers.GRUCell(units, dropout=0.5)
        self.rnn_cell1 = layers.GRUCell(units, dropout=0.5)

        ## 全连接层 预测结果 [b, 128] ==> [b, 1]  好评与坏评
        self.outlayer = layers.Dense(1)

    #堆叠网络
    def call(self, inputs, training=None):
        # inputs:[b, 80]   training:默认为 训练模式
        # 输入层
        x = inputs
        #print("x",x)
        #输入数据经过嵌入层进行词嵌入
        x = self.embedding(x)
        #print("x1",x)
        #记忆参数
        state0 = self.state0
        state1 = self.state1

        # 这里将文本按照第二个维度展开 并将每个词映射成一个100维的向量,循环按照第二个维度一个一个词迭代
        # batch个句子 循环取第i个单词（80个单词）处理
        for word in tf.unstack(x, axis=1): # word[b, 100]  [b, axis=1, 100]
            #print("word",word)
            # h1 = x* wxh + h0* whh
            out0, state0 = self.rnn_cell0(word, state0, training)
            #print("state0:",state0)
            #print("out0:",out0)
            out1, state1 = self.rnn_cell1(out0, state1, training)
            #print("state1:", state1)
            #print("out1:", out1)

        #输出层  out：[b, 128] ==> [b, 1]
        x = self.outlayer(out1)
        #print("x2",x)
        #概率输出
        prob = tf.sigmoid(x)

        return prob


def main():
    units = 64 #循环神经网络 宽度
    epochs = 1 #总循环次数

    import time

    t0 = time.time()

    #创建对象
    model = MyRNN(units)
    #编译模型
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    #训练模型
    model.fit(db_train, epochs=1, validation_data=db_test)

    #评估模型
    model.evaluate(db_test)
    #print("predict",model.predict(de_bb))
    #print(y_test)
    model.summary()
    plot_model(model, to_file="model.png", show_shapes=True)
    #model.load_weights('model_weight.h5')
    #model.evaluate(db_test)
    #model.save_weights('model_weight.h5')

    t1 = time.time()

    print('total time cost:', t1 - t0)


if __name__ == "__main__":
    main()



