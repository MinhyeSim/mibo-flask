from keras import backend as K
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from matplotlib.pyplot import cla


class Solution():
    def __init__(self) -> object:  
        self.char_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
                'h', 'i', 'j', 'k', 'l', 'm', 'n',
                'o', 'p', 'q', 'r', 's', 't', 'u',
                'v', 'w', 'x', 'y', 'z']

        self.num_dic = {n: i for i, n in enumerate(self.char_arr)}
        self.dic_len = len(self.num_dic)
        self.seq_data = ['word', 'wood', 'deep', 'dive', 'cold', 'cool', 'load', 'love', 'kiss', 'kind']
        self.optimizer = None
        self.states = None
        self.cost = None
        self.model1 = None
        self.input_batch = []
        self.target_batch = []
        self.X = None
        self.Y = None

    def hook(self):
        self.make_batch()
        self.model()
        self.train()


    def make_batch(self):
        input_batch = self.input_batch
        target_batch = self.target_batch
        num_dic = self.num_dic 
        

        for seq in self.seq_data:
            input = [num_dic[n] for n in seq[:-1]]
            target = num_dic[seq[-1]]  # -1 은 all
            input_batch.append(np.eye(self.dic_len)[input])
            target_batch.append(target)
        return input_batch, target_batch

 
    def model(self):
        learning_rate = 0.01
        n_hidden = 128        
        n_step = 3
        # 타입스텝: [1, 2, 3] => 3
        # RNN 을 구성하는 시퀀스의 갯수
        n_input = n_class = self.dic_len
        # 입력값 크기. 알파벳에 대한 ohe 이므로 26개가 됨
        # 따라서 c 를 선택하면 [0 0 1 0 0 0 .....0]
        # 출력값도 입력값과 마찬가지로 26개의 알파벳으로 분류합니다.
        n_input = n_class = self.dic_len      

        # *******
        # 신경망 모델 구성
        # *******

        X = tf.placeholder(tf.float32, [None, n_step, n_input])
        Y = tf.placeholder(tf.int32, [None])

        W = tf.Variable(tf.random_normal([n_hidden, n_class]))
        b = tf.Variable(tf.random_normal([n_class]))

        cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
        # RNN 셀 생성
        cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
        # 과적합 방지를 위한 Dropout 기법을 사용
        cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
        # 여러개의 셀을 조합하기 위해 추가 셀 생성
        multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
        # 여러개의 셀을 조합한 RNN 셀을 생성

        outputs, self.states = tf.nn.dynamic_rnn(multi_cell, X, dtype = tf.float32)
        #  tf.nn.dynamic_rnn 을 이용해 순환 신경망을 생성

        # 최종 결과는 ohe 형식으로 생성
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = outputs[-1]
        model1 = tf.matmul(outputs, W) + b

        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=model1, labels=Y
        ))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    def train(self):
        total_epoch = 30 # 훈련횟수
        # *******
        # 신경망 모델 학습
        # *******

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        input_batch, target_batch = self.make_batch()
        for epoch in range(total_epoch):
            _, loss = sess.run([self.optimizer, self.cost],
                            {self.X: input_batch, self.Y: target_batch})
            print("Epoch: ", "%04d" % (epoch + 1),
                "cost: ", "{:.6f}".format(loss))
        print('===최적화 완료===')

        # *******
        # 신경망 모델 검증
        # *******
        K.clear_session()
        prediction = tf.cast(tf.argmax(self.model1, 1), tf.int32)
        prediction_check = tf.equal(prediction, self.Y)
        # 문자열 값비교 equal
        accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))
        input_batch, target_batch = self.make_batch()

        predict, accuracy_val = sess.run([prediction, accuracy],
                                        {self.X: input_batch, self.Y: target_batch})

        predict_words = []
        for idx, val in enumerate(self.seq_data):
            last_char = self.char_arr[predict[idx]]
            predict_words.append(val[:3] + last_char)

        print('\n ===== 예측결과 ====')
        print('입력값: ', [W[:3] + ' ' for W in self.seq_data])
        print('예측값: ', predict_words)
        print('정확도: ', accuracy_val)
if __name__ == '__main__' : 

    Solution().hook()