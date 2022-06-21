# 날씨 정보를 이용해 양배추 가격 예측하기
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import basedir
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Solution():
    def __init__(self) -> object:
       self.model = os.path.join(basedir, 'model')


    def hook(self):
        self.read()
        self.processing()
        self.create_model()

    def read(self):
        df = pd.read_csv('./data/price_data.csv', encoding='UTF-8')
        print (df)
        df.to_csv('./save/price_data.csv', index=False)

    def processing(self):
        pass
    
    def create_model(self): #모델생성
        sess = tf.Session()
        _ = tf.Variable(initial_value = 'fake_variable')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(self.model, 'cabbage', 'model'), global_step=1000)

        

    
    

if __name__=='__main__':
    Solution().hook()