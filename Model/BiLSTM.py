import tensorflow as tf

BasicLSTMCell, DropoutWrapper, MultiRNNCell = tf.nn.rnn_cell.BasicLSTMCell, tf.nn.rnn_cell.DropoutWrapper, tf.nn.rnn_cell.MultiRNNCell


class BiLSTM(object):
    def __init__(self, lstm_size=128, layers=2, wv_dim=100, vsize=100, trunc_norm_init_std=1e-4):
        self._lstm_size = lstm_size
        self._layers = layers
        self._wv_dim = wv_dim
        self._vsize = vsize
        self._trunc_norm_init_std = trunc_norm_init_std
        self._build_graph()

    def _build_graph(self):
        with tf.name_scope('input'):
            with tf.variable_scope('embedding'):
                embedding = tf.get_variable('embedding', [self._vsize, self._wv_dim], dtype=tf.float64, initializer=tf.truncated_normal_initializer(stddev=self._trunc_norm_init_std))
                embedded_x = tf.nn.embedding_lookup(embedding, self._x) 

            with tf.name_scope('multi-bilstm-layer'):
                fw_cells = [DropoutWrapper(BasicLSTMCell(self._lstm_size, state_is_tuple=True),
                                       output_keep_prob=self._drop_out_placeholder) for _ in range(self._layers)]
                bw_cells = [DropoutWrapper(BasicLSTMCell(self._lstm_size, state_is_tuple=True),
                                       output_keep_prob=self._drop_out_placeholder) for _ in range(self._layers)]
                self._fw_cell = MultiRNNCell(fw_cells)
                self._bw_cell = MultiRNNCell(bw_cells)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(self._fw_cell, self._bw_cell, embedded_x, sequence_length=self._lens, dtype=tf.float64)
            output = tf.concat(outputs, axis=2)
            self._output = output

    def io_nodes(self):
        return self._x, self._lens, self._drop_out_placeholder, self._output


if __name__ == "__main__":
    lstm = BiLSTM()
