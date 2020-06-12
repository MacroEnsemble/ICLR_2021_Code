import numpy as np
import tensorflow as tf
import random
import os

class Estimator():
    """Q-Value Estimator neural network.
    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, scope="estimator", summaries_dir=None, lr=None, LEN_SKILL=None, VALID_ACTIONS=None):
        self.lr = lr
        self.LEN_SKILL=LEN_SKILL
        self.VALID_ACTIONS=VALID_ACTIONS
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """

        # Placeholders for our input
        # Our input are 4 RGB frames of shape 160, 160 each
        # self.X_pl = tf.placeholder(shape=[None, LEN_SKILL, len(VALID_ACTIONS), 4], dtype=tf.int32, name="X")
        self.X_pl = tf.placeholder(shape=[None, self.LEN_SKILL, len(self.VALID_ACTIONS)], dtype=tf.int32, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        batch_size = tf.shape(self.X_pl)[0]

        X = tf.dtypes.cast(self.X_pl, tf.float32)
        # Fully connected layers
        flattened = tf.contrib.layers.flatten(X)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        self.predictions = tf.contrib.layers.fully_connected(fc1, len(self.VALID_ACTIONS), activation_fn=None)

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.abs_loss = tf.reduce_mean(
                tf.losses.absolute_difference(self.y_pl, self.action_predictions))
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(self.lr, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.scalar("abs_loss", self.abs_loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions)),
            tf.summary.scalar("min_q_value", tf.reduce_min(self.predictions)),
            tf.summary.scalar("avg_q_value", tf.reduce_mean(self.predictions))
        ])


    def predict(self, sess, s):
        return sess.run(self.predictions, { self.X_pl: s})

    def update(self, sess, s, a, y):
        """
        Updates the estimator towards the given targets.

        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 4, len_skill, action_space]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss

    def get_abs_loss(self, sess, s, a, y):
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
        abs_loss = sess.run(self.abs_loss, feed_dict)
        return abs_loss