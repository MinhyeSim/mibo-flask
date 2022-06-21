import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import basedir
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from icecream import ic


class Solution():
    def __init__(self) -> None:
        self.model = os.path.join(basedir, 'cabbage_model')
        self.df = None
    
    def hook(self):
        self.preprocess()
        self.create_model()

    def preprocess(self):
        self.df = pd.read_csv('./data/price_data.csv')
        xy = np.array(self.df,  dtype=np.float32)
        ic(type(xy))

    def create_model(self):
        sess = tf.Session()
        _ = tf.Variable(initial_value = 'fake_variable')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(self.model, 'cabbage.ckpt'), global_step=1000) 
        print('저장완료')

if __name__ == "__main__":
    Solution().hook()