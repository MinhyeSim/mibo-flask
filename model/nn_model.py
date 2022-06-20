import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import basedir
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

class Solution():
    def __init__(self) -> None:
        pass
        

    def create(self):
        # *******
        # 신경망 모델 구성
        # *******
        X = tf.placeholder(tf.float32)
        Y = tf.placeholder(tf.float32)

        W = tf.Variable(tf.random_uniform([2, 3], -1, 1.))
        # 신경망 neural network 앞으로는 nn 으로 표기
        # nn 은 2차원으로 [입력층(특성), 출력층(레이블)] -> [2, 3] 으로 정합니다

        b = tf.Variable(tf.zeros([3]))
        # b 는 편향. 앞으로 편향은 b 로 표기
        # W 는 가중치. 앞으로는 가중치는 W 로 표기
        # b 는 각 레이어의 아웃풋 갯수로 설정함.
        # b 는 최종 결과값의 분류 갯수인 3으로 설정함.

        L = tf.add(tf.matmul(X, W), b)
        # 가중치와 편향을 이용해 계산한 결과 값에
        L = tf.nn.relu(L)
        # TF 에서 기본적으로 제공하는 활성화 함수인 ReLU 함수를 적용

        model = tf.nn.softmax(L)
        # softmax() 를 사용해서 출력값을 사용하기 쉽게 만듦
        # 소프트맥스 함수는 다음처럼 결과값을 전체합이 1인 확률로 만들어주는 함수
        # 예) [8.04, 2.76, -6.52] -> [0.53, 0.24, 0.23]

        cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis = 1))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(cost)

        # 비용함수를 최소화 시키면 -> 경사도를 0로 만들어 가면 그 값이 최적화된 값일 것이다...

    def fit(self):
        train_op = self.train_op
        # **********
        # 신경망 학습 모델
        # **********
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        for step in range(100):
            sess.run(train_op, {X: x_data, Y: y_data})
            if (step + 1) % 10 == 10:
                print(step +1, sess.run(cost, {X: x_data, Y: y_data}))
    def eval(self):
        model = self.model


        # *********
        # 결과확인
        # ********
        prediction = tf.argmax(model, 1)
        target = tf.argmax(Y, 1)
        print('예측값', sess.run(prediction, {X: x_data}))
        print('실제값', sess.run(target, {Y: y_data}))
        # tf.argmax: 예측값과 실제값의 행렬에서 tf.argmax 를 이용해 가장 큰 값을 가져옴
        # 예) [[0, 1, 0][1, 0, 0]] -> [1, 0]
        #  [[0.2, 0.7, 0.1][0.9, 0.1, 0.]] -> [1, 0]
        is_correct = tf.equal(prediction, target)
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        print('정확도: %.2f' % sess.run(accuracy * 100, {X: x_data, Y: y_data}))

if __name__=='__main__':
    # [털, 날개]
        x_data = np.array(
            [[0, 0],[1, 0],[1, 1],[0, 0],[0, 0], [0, 1]]
        )

        #기타, 포유류, 조류
        # 원핫 인코딩
        y_data = np.array([
            [1, 0, 0], # 기타
            [0, 1, 0], # 포유류
            [0, 0, 1], # 조류
            [1, 0, 0], # 기타
            [0, 1, 0], # 포유류
            [0, 0, 1] # 조류
        ])
        tf.disable_v2_behavior()

        s = Solution()
        s.create()
        s.fit()
        s.eval()