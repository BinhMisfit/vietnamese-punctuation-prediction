import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell, LSTMCell

def conv1d(in_, filter_size, height, padding, is_train=True, drop_rate=0.0, scope=None):
    with tf.variable_scope(scope or "conv1d"):
        num_channels = in_.get_shape()[-1]
        filter_ = tf.get_variable("filter", shape=[1, height, num_channels, filter_size], dtype=tf.float32)
        bias = tf.get_variable("bias", shape=[filter_size], dtype=tf.float32)
        strides = [1, 1, 1, 1]
        in_ = tf.layers.dropout(in_, rate=drop_rate, training=is_train)
        # [batch, max_len_sent, max_len_word / filter_stride, char output size]
        xxc = tf.nn.conv2d(in_, filter_, strides, padding) + bias
        out = tf.math.reduce_max(tf.nn.relu(xxc), axis=2)  # max-pooling, [-1, max_len_sent, char output size]
        return out


def multi_conv1d(in_, filter_sizes, heights, padding="VALID", is_train=True, drop_rate=0.0, scope=None):
    with tf.variable_scope(scope or "multi_conv1d"):
        assert len(filter_sizes) == len(heights)
        outs = []
        for i, (filter_size, height) in enumerate(zip(filter_sizes, heights)):
            if filter_size == 0:
                continue
            out = conv1d(in_, filter_size, height, padding, is_train=is_train, drop_rate=drop_rate,
                         scope="conv1d_{}".format(i))
            outs.append(out)
        concat_out = tf.concat(axis=2, values=outs)
        return concat_out


class AttentionCell(RNNCell):
    def __init__(self, num_units, memory, pmemory, cell_type='lstm'):
        super(AttentionCell, self).__init__()
        self._cell = LSTMCell(num_units)
        self.num_units = num_units
        self.memory = memory
        self.pmemory = pmemory
        self.mem_units = memory.get_shape().as_list()[-1]

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def compute_output_shape(self, input_shape):
        pass

    def __call__(self, inputs, state, scope=None):
        # Attention model
        c, m = state # c is previous cell state, m is previous hidden state
        ha = tf.nn.tanh(tf.add(self.pmemory, tf.layers.dense(m, self.mem_units, use_bias=False, name="wah")))
        alphas = tf.squeeze(tf.exp(tf.layers.dense(ha, units=1, use_bias=False, name='way')), axis=[-1])
        alphas = tf.math.divide(alphas, tf.math.reduce_sum(alphas, axis=0, keepdims=True))  # (max_time, batch_size)

        w_context = tf.math.reduce_sum(tf.multiply(self.memory, tf.expand_dims(alphas, axis=-1)), axis=0)
        h, new_state = self._cell(inputs, state)
        # Late fusion
        lfc = tf.layers.dense(w_context, self.num_units, use_bias=False, name='wfc')  # late fused context

        fw = tf.sigmoid(tf.layers.dense(lfc, self.num_units, use_bias=False, name='wff') +
                        tf.layers.dense(h, self.num_units, name='wfh')) # fusion weights
        
        hft = tf.math.multiply(lfc, fw) + h  # weighted fused context + hidden state
        
        return hft, new_state
