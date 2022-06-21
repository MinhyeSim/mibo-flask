# 날씨 정보를 이용해 양배추 가격 예측하기
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import basedir
from icecream import ic

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Solution():
    def __init__(self) -> None:
       self.model = os.path.join(basedir, 'cab_model')
       self.df = None
       self.x_data = None
       self.y_data = None


    def hook(self):
        self.processing()
        

    def processing(self): #모델생성
        self.df = pd.read_csv('./data/price_data.csv', encoding='UTF-8', thousands=',')
        
        # year,avgTemp,minTemp,maxTemp,rainFall,avgPrice
        xy = np.array(self.df, dtype=np.float32) #csv파일을 배열로 변환
        self.x_data = xy[:,1:-1]
        self.y_data = xy[:,[-1]]
        ic(self.x_data, self.y_data)

    def create_model(self):
        
        # 텐서 모델 초기화(모델템플릿 생성)
        model = tf.global_variables_initializer()
        # 확률변수 데이터 
        self.processing()
        # 선형식(가설) 제작 y = Wx+b
        X = tf.placeholder(tf.float32, shape=[None, 4])
        Y = tf.placeholder(tf.float32, shape=[None, 1])
        W = tf.Variable(tf.random_normal([4, 1]), name="weight") # 변하는 값
        b = tf.Variable(tf.random_normal([1]), name="bias")
        #가설 -> 선형식
        hypothesis = tf.matmul(X,W) + b 
        
        # 손실함수
        cost = tf.reduce_mean(tf.square(hypothesis - Y))
        # 최적화 알고리즘
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)
        train = optimizer.minimize(cost)
        # 세션생성
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        #트레이닝
        for step in range(10000):
            cost_, hypo_, _ = sess.run([cost, hypothesis, train],
                                        feed_dict={X: self.x_data, Y: self.y_data})
            if step % 500 == 0 :
                print('# %d 손실비용: %d' %(step, cost_))
                print('배추가격 : %d' %(hypo_[0]))                                
        #모델저장
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(self.model, 'cabbage.ckpt'), global_step=1000) 
        print('저장완료')
    
    def load_model(self): #모델로드
         # 선형식(가설) 제작 y = Wx_b
        X = tf.placeholder(tf.float32, shape=[None, 4])
        W = tf.Variable(tf.random_normal([4, 1]), name="weight") # 변하는 값
        b = tf.Variable(tf.random_normal([1]), name="bias")
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, '')


        

if __name__=='__main__':
    #Solution().processing()
    Solution().create_model()