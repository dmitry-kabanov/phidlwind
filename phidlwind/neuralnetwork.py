import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import tensorflow.keras as keras  # noqa: E402

from typing import Dict, List  # noqa: E402


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

        tf.compat.v1.enable_eager_execution()

        self.lb = np.array([np.min(self.x1), np.min(self.x2)])
        self.ub = np.array([np.max(self.x1), np.max(self.x2)])

        self.X_train = 2 * (X - self.lb) / (self.ub - self.lb) - 1.0

        assert np.all(self.X_train >= -1.0)
        assert np.all(self.X_train <= +1.0)

        x1 = np.linspace(-1, 1, num=64)
        x2 = np.linspace(-1, 1, num=64)
        xx1, xx2 = np.meshgrid(x1, x2)
        xx1_col, xx2_col = xx1.flatten()[:, None], xx2.flatten()[:, None]
        grid = np.hstack((xx1_col, xx2_col))
        self.X_pnlt = tf.convert_to_tensor(grid, dtype=tf.float32)
        self._pnlt_grid_size = (len(x1), len(x2))

        self._indices = ([], [0])

        self.history: Dict[str, List] = {}

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
        if y_pred.shape[0]:
            u1_true = y_true[:, 0]
            u1_pred = y_pred[:, 0]
            mse1 = tf.reduce_mean(tf.square(u1_true - u1_pred))
            u2_true = y_true[:, 1]
            u2_pred = y_pred[:, 1]
            mse2 = tf.reduce_mean(tf.square(u2_true - u2_pred))
        else:
            mse1 = tf.contrib.eager.Variable(0.0, dtype=tf.float32)
            mse2 = tf.contrib.eager.Variable(0.0, dtype=tf.float32)

        idx = np.asarray(self._indices[1]) - self.X_train.shape[0]
        pnlt_pts = tf.gather(self.X_pnlt, idx)
        with tf.GradientTape() as t:
            # Evaluate model on the constraint grid.
            t.watch(pnlt_pts)
            result = self.model(pnlt_pts)
        derivs = t.gradient(result, pnlt_pts)

        # derivs = tf.gradients(y_pred, self.model.input)[0]
        du1_dx1 = derivs[:, 0]
        du2_dx2 = derivs[:, 1]

        f = du1_dx1 + du2_dx2

        pnlt = self.gamma * tf.reduce_mean(tf.square(f))

        loss = mse1 + mse2 + pnlt

        return loss

    def train(self, num_epochs: int):
        model = self.model

        # Save training history.
        self.history = {"epoch_loss_avg": []}

        for epoch in range(1, num_epochs+1):
            epoch_loss_avg = tf.keras.metrics.Mean()

            # Training loop - using batches of 32.
            for indices, x, y in self._get_batch(32):
                # Optimize the model.
                loss_value, gradients = self._grad(indices, x, y)
                vars = model.trainable_variables
                model.optimizer.apply_gradients(zip(gradients, vars))

                # Track progress.
                epoch_loss_avg(loss_value)  # Add current batch loss

            # Actions at the end of an epoch.
            self.history["epoch_loss_avg"].append(epoch_loss_avg.result())
            print("Epoch {:03d}: Loss: {:.3e}".format(epoch, epoch_loss_avg.result()))

    def _get_batch(self, batch_size):
        N = self.X_train.shape[0]
        P = self.X_pnlt.shape[0]
        indices_all = list(range(0, N+P))
        np.random.shuffle(indices_all)
        indices_all_shuffled = indices_all

        lower = 0
        indices_batch = []

        while lower < N+P:
            upper = min(lower+batch_size, N+P)
            indices_batch = indices_all_shuffled[lower : upper]
            indices_batch.sort()

            idx_misfit = []
            idx_penalty = []
            X_batch = []
            y_batch = []
            for idx in indices_batch:
                if idx < N:
                    idx_misfit.append(idx)
                    X_batch.append(self.X_train[idx])
                    y_batch.append(self.y_train[idx])
                else:
                    idx_penalty.append(idx)

            assert len(idx_misfit) + len(idx_penalty) <= batch_size
            assert len(y_batch) <= batch_size
            assert len(X_batch) == len(y_batch)
            yield (idx_misfit, idx_penalty), X_batch, y_batch
            lower = upper

    def _grad(self, indices, inputs, targets):
        self._indices = indices
        inputs = np.asarray(inputs)
        targets = np.asarray(targets)
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        targets = tf.convert_to_tensor(targets, dtype=tf.float32)
        model = self.model
        with tf.GradientTape() as tape:
            if inputs.shape[0].value:
                y_pred = model(inputs)
            else:
                y_pred = tf.contrib.eager.Variable([], shape=(None,), dtype=tf.float32)
            vars = model.trainable_variables
            loss_value = self._compute_loss(targets, y_pred)
            gradients = tape.gradient(loss_value, vars)
            gradients = [g if g is not None else tf.zeros_like(v)
                         for g, v in zip(gradients, vars)]
            return loss_value, gradients

    def predict(self, X_new):
        X_new_tilde = 2 * (X_new - self.lb) / (self.ub - self.lb) - 1.0
        return self.model.predict(X_new_tilde)

    def get_physical_constraint(self):
        pnlt_pts = self.X_pnlt
        with tf.GradientTape() as t:
            # Evaluate model on the constraint grid.
            t.watch(pnlt_pts)
            result = self.model(pnlt_pts)
        derivs = t.gradient(result, pnlt_pts)

        # derivs = tf.gradients(y_pred, self.model.input)[0]
        du1_dx1 = derivs[:, 0]
        du2_dx2 = derivs[:, 1]

        f = du1_dx1 + du2_dx2

        f_result = f.numpy().reshape(self._pnlt_grid_size)

        return f_result
