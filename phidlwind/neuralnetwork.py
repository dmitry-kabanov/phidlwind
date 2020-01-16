import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class DivFreeNeuralNetwork:
    """Neural network that models divergence-free velocity field."""

    def __init__(self, X, y, gamma):
        # Scalar of the penalty term.
        self.gamma = gamma

        # Notice that the following subtle broadcasting rule is used:
        # slice X[:, 0:1] returns a 2D array, in the opposite to X[:, 0] which
        # returns a 1D array.
        self.x1 = X[:, 0:1]
        self.x2 = X[:, 1:2]
        self.u1 = y[:, 0:1]
        self.u2 = y[:, 1:2]

        self.X_train = X
        self.y_train = y

        self.lb = np.array([np.min(self.x1), np.min(self.x2)])
        self.ub = np.array([np.max(self.x1), np.max(self.x2)])

        self.X_train = 2 * (X - self.lb) / (self.ub - self.lb) - 1.0

        assert np.all(self.X_train >= -1.0)
        assert np.all(self.X_train <= +1.0)

        N = X.shape[0]
        x1 = np.linspace(-1, 1, num=10 * N)
        x2 = np.linspace(-1, 1, num=10 * N)
        xx1, xx2 = np.meshgrid(x1, x2)
        self._grid = np.hstack((xx1.flatten()[:, None], xx2.flatten()[:, None]))
        self._grid = tf.convert_to_tensor(self._grid, dtype=tf.float32)

        # TODO: Check how variables are initialized.
        self.model = keras.models.Sequential(
            [
                keras.layers.InputLayer(input_shape=(2,)),
                keras.layers.Dense(
                    20, activation="tanh", kernel_initializer="glorot_uniform"
                ),
                keras.layers.Dense(
                    10, activation="tanh", kernel_initializer="glorot_uniform"
                ),
                keras.layers.Dense(2),
            ]
        )

        self.model.compile(loss=self._compute_loss, optimizer="sgd")

    def _compute_loss(self, y_true, y_pred):
        u1_true = y_true[:, 0]
        u2_true = y_true[:, 1]
        u1_pred = y_pred[:, 0]
        u2_pred = y_pred[:, 1]

        with tf.GradientTape() as t:
            # Evaluate model on the constraint grid.
            t.watch(self._grid)
            result = self.model(self._grid)
        derivs = t.gradient(result, self._grid)

        # derivs = tf.gradients(y_pred, self.model.input)[0]
        du1_dx1 = derivs[:, 0]
        du2_dx2 = derivs[:, 1]

        f = du1_dx1 + du2_dx2

        mse1 = tf.reduce_mean(tf.square(u1_true - u1_pred))
        mse2 = tf.reduce_mean(tf.square(u2_true - u2_pred))
        pnlt = self.gamma * tf.reduce_mean(tf.square(f))

        loss = mse1 + mse2 + pnlt

        return loss

    def train(self, epochs):
        self.history = self.model.fit(self.X_train, self.y_train, epochs=epochs)
        # tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u}

        # start_time = time.time()
        # for it in range(nIter):
        #     self.sess.run(self.train_op_Adam, tf_dict)

        #     # Print
        #     if it % 100 == 0:
        #         elapsed = time.time() - start_time
        #         loss_value = self.sess.run(self.loss, tf_dict)
        #         lambda_value = self.sess.run(self.lambda_)
        #         print('It: %d, Loss: %.3e, Lambda: %.3f, Time: %.2f' %
        #               (it, loss_value, lambda_value, elapsed))
        #         start_time = time.time()

        # fetches = [self.loss, self.lambda_]
        # self.optimizer.minimize(self.sess,
        #                         feed_dict=tf_dict,
        #                         fetches=fetches,
        #                         loss_callback=self.callback)
        # loss_final = self.sess.run(self.loss, feed_dict=tf_dict)
        # lambda_final = self.get_pde_params()[0]
        # print('=== Loss: %e, lambda: %.5f' % (loss_final, lambda_final))
        # print('=== # of iterations ', self._counter)

    def predict(self, X_new):
        X_new_tilde = 2 * (X_new - self.lb) / (self.ub - self.lb) - 1.0
        return self.model.predict(X_new_tilde)
