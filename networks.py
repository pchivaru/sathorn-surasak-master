import keras
from keras.models import Model
from keras.layers import Input, Dense, LeakyReLU, Flatten
from keras.layers.convolutional import Conv2D
from keras import regularizers
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import numpy as np
from scipy.special import softmax


CLIPRANGE = 0.2
E_MARGIN = 0.8

keras.losses.huber_loss = tf.losses.Huber


class DesignedLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DesignedLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Constrain > 0
        self.w = self.add_weight(name='weights', shape=input_shape[1:], initializer='ones', trainable=True,
                                 constraint=lambda x: tf.clip_by_value(x, 1e-8, np.inf))
        self.p = self.add_weight(name='exponents', shape=input_shape[1:], initializer='ones', trainable=True,
                                 constraint=lambda x: tf.clip_by_value(x, 1e-8, np.inf))
        super(DesignedLayer, self).build(input_shape)

    def call(self, x):
        x = x + 1e-8    # partial deriv includes log
        x = tf.multiply(x, self.w)
        x = tf.math.pow(x, self.p)

        state = x[:, :, :-4]
        modif = x[:, :, -4:]
        state = tf.reduce_sum(state, axis=-1)
        modif = tf.reduce_sum(modif, axis=-1)
        out = tf.multiply(state, modif)
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]


keras.layers.DesignedLayer = DesignedLayer


class DQNNet(object):

    def __init__(self, input_shape, num_actions, learning_rate):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.model = keras.Sequential()
        self.model.add(Conv2D(64, input_shape=self.input_shape, kernel_size=2))
        self.model.add(LeakyReLU())
        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(LeakyReLU())
        self.model.add(Dense(64))
        self.model.add(LeakyReLU())
        self.model.add(Dense(self.num_actions))
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='huber_loss')

    def save(self, file):
        self.model.save(file)

    def load(self, file):
        self.model = keras.models.load_model(file)

    # Returns the best action given a single state
    def best_action(self, state):
        predicted = self.model.predict(np.asarray([state]))[0]
        action = np.argmax(predicted)
        return action

    def softmax_values(self, states):
        predicted = self.model.predict_on_batch(states)
        predicted = softmax(predicted, axis=-1)
        return predicted

    # Returns an array of best action values given an array of states
    def best_value(self, states):
        predicted = self.model.predict_on_batch(states)
        return np.max(predicted, axis=-1)

    def best_actions(self, states):
        predicted = self.model.predict_on_batch(states)
        return np.argmax(predicted, axis=-1)

    def get_actions_values(self, states, actions):
        predicted = self.model.predict_on_batch(states)
        values = np.empty(shape=len(states))
        for i in range(len(states)):
            values[i] = predicted[i][actions[i]]
        return values

    # Fits network to a batch of transitions while updating only the observed action
    def train(self, states, actions, utilities):
        labels = self.model.predict_on_batch(states)
        for i in range(len(states)):
            labels[i][actions[i]] = utilities[i]
        self.model.train_on_batch(states, labels)



class DQNPoly(object):
    def __init__(self, input_shape, num_actions, learning_rate):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate

        states = Input(shape=self.input_shape)
        ply_layer = DesignedLayer()(states)

        self.model = Model(inputs=states, outputs=ply_layer)
        self.model.compile(optimizer=Adam(self.learning_rate), loss='huber_loss')

    def update_learning_rate(self, learning_rate):
        K.set_value(self.model.optimizer.lr, learning_rate)

    def save(self, file):
        print(file)
        self.print_parameters()
        self.model.save(file)

    def load(self, file):
        self.model = keras.models.load_model(file)
        self.print_parameters()
        print(file)

    def best_action(self, state):
        predicted = self.model.predict(np.asarray([state]))[0]
        action = np.argmax(predicted)
        return action

    def best_value(self, states):
        predicted = self.model.predict_on_batch(states)
        return np.max(predicted, axis=-1)

    def values(self, state):
        predicted = self.model.predict(np.asarray([state]))[0]
        return predicted

    def print_parameters(self):
        parameters = self.model.layers[1].get_weights()
        w = parameters[0]
        p = parameters[1]
        for i in range(self.input_shape[0]):
            act = ''
            for j in range(self.input_shape[1]):
                act += str(w[i][j])+', '
            print(act)
        print('---')
        for i in range(self.input_shape[0]):
            act = ''
            for j in range(self.input_shape[1]):
                act += str(p[i][j])+', '
            print(act)

    def train(self, states, actions, utilities):
        labels = self.model.predict_on_batch(states)
        for i in range(len(states)):
            labels[i][actions[i]] = utilities[i]

        self.model.train_on_batch(states, labels)


class DQNPolyMax(object):
    def __init__(self, input_shape, num_actions, learning_rate):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate

        states = Input(shape=self.input_shape)
        ply_layer = DesignedLayer()(states)
        out = keras.layers.Softmax()(ply_layer)

        self.model = Model(inputs=states, outputs=out)
        self.model.compile(optimizer=Adam(self.learning_rate), loss=keras.losses.categorical_crossentropy)

    def update_learning_rate(self, learning_rate):
        K.set_value(self.model.optimizer.lr, learning_rate)

    def save(self, file):
        print(file)
        self.print_parameters()
        self.model.save(file)

    def load(self, file):
        self.model = keras.models.load_model(file)
        self.print_parameters()
        print(file)

    def best_action(self, state):
        predicted = self.model.predict(np.asarray([state]))[0]
        action = np.argmax(predicted)
        return action

    def best_actions(self, states):
        predicted = self.model.predict_on_batch(states)
        return np.argmax(predicted, axis=-1)

    def values(self, state):
        predicted = self.model.predict(np.asarray([state]))[0]
        return predicted

    def print_parameters(self):
        parameters = self.model.layers[1].get_weights()
        w = parameters[0]
        p = parameters[1]
        for i in range(self.input_shape[0]):
            act = ''
            for j in range(self.input_shape[1]):
                act += str(w[i][j]) + ', '
            print(act)
        print('---')
        for i in range(self.input_shape[0]):
            act = ''
            for j in range(self.input_shape[1]):
                act += str(p[i][j]) + ', '
            print(act)

    def train(self, states, actions):
        self.model.train_on_batch(states, actions)
