import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt


class Solution():
    def __init__(self) -> None:
        self.X = None
        self.Y = None
        self.W_history = []
        self.cost_history = []

    def hook(self):
        self.main()
        self.chart()


    def main(self):        

        tf.set_random_seed(777)

        self.X = [1, 2, 3]
        self.Y = [1, 2, 3]

        W = tf.placeholder(tf.float32)
        hypothesis = self.X * W

        cost = tf.reduce_mean(tf.square(hypothesis - self.Y))
        sess = tf.Session()



        for i in range(-30, 50):
            curr_W = i * 0.1
            curr_cost = sess.run(cost, {W: curr_W})
            self.W_history.append(curr_W)
            self.cost_history.append(curr_cost)
        # 차트로 확인
    def chart(self):
        plt.plot(self.W_history, self.cost_history)
        plt.show()

if __name__=='__main__':
    Solution().hook()