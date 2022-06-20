
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import basedir
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

class Solution:
    def __init__(self) :
        self.x_data = np.array([[0, 0],[1, 0],[1, 1],[0, 0],[0, 0], [0, 1]])
        self.y_data = np.array([
            [1, 0, 0], # 기타
            [0, 1, 0], # 포유류
            [0, 0, 1], # 조류
            [1, 0, 0], # 기타
            [0, 1, 0], # 포유류
            [0, 0, 1] # 조류
        ])
        self.train_op = None
        self.X = None
        self.Y = None
        self.W = None
        self.b = None
        self.L = None
        self.cost = None
        self.model = None


    def hook(self): #정의, 훈련, 실행시키는 메소들로 나뉠 예정. 최소 3개
        
        self.create_nn_model()
        self.train_nn_model()

    def create_nn_model(self) :
        self.X = tf.placeholder(tf.float32)
        self.Y = tf.placeholder(tf.float32)
        self.W = tf.Variable(tf.random_uniform([2, 3], -1, 1.))
        self.b = tf.Variable(tf.zeros([3]))
        self.L = tf.add(tf.matmul(self.X, self.W), self.b)
        self.L = tf.nn.relu(self.L)
        self.model = tf.nn.softmax(self.L)
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.Y * tf.log(self.model), axis = 1))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        self.train_op = optimizer.minimize(self.cost)
       
    def train_nn_model(self): 
        
        sess = tf.Session() 
        sess.run(tf.global_variables_initializer())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
        

            for step in range(100):
                sess.run(self.train_op, {self.X: self.x_data, self.Y: self.y_data})
                if (step + 1) % 10 == 10:
                    print(step +1, sess.run(self.cost, {self.X: self.x_data, self.Y: self.y_data}))
            

            prediction = tf.argmax(self.model, 1)
            target = tf.argmax(self.Y, 1)
            print('예측값', sess.run(prediction, {self.X: self.x_data}))
            print('실제값', sess.run(target, {self.Y:self.y_data}))
            # tf.argmax: 예측값과 실제값의 행렬에서 tf.argmax 를 이용해 가장 큰 값을 가져옴
            # 예) [[0, 1, 0][1, 0, 0]] -> [1, 0]
            #  [[0.2, 0.7, 0.1][0.9, 0.1, 0.]] -> [1, 0]
            is_correct = tf.equal(prediction, target)
            accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
            print('정확도: %.2f' % sess.run(accuracy * 100, {self.X: self.x_data, self.Y: self.y_data}))

if __name__ == '__main__' : 
    Solution().hook()