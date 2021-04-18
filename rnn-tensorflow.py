import tensorflow as tf

class RNNTensor(tf.keras.layers.Layer):
    def __init__(self, rnn_units, input_dim,output_dim):
        super(RNNTensor, self).__init__()
    #Initialize the weights
        self.wxh = self.add_weight([rnn_units, input_dim])
        self.whh = self.add_weight([rnn_units, rnn_units])
        self.wyh = self.add_weight([output_dim, rnn_units])
    #initialize hidden state to zero
        self.h = tf.zeros([rnn_units,1])
    def call(self,x):
        #update teh hidden state
        self.h = tf.math.tanh(self.whh * self.h + self.wxh * self.x)
        #compute the output
        output = self.wyh * self.h
        #return the output and hidden state
        return output, self.h


