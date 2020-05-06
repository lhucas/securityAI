import tensorflow as tf
from tensorflow.contrib import slim



BasicLSTMCell, DropoutWrapper, MultiRNNCell = tf.nn.rnn_cell.BasicLSTMCell, tf.nn.rnn_cell.DropoutWrapper, tf.nn.rnn_cell.MultiRNNCell

if __name__ == "__main__":
#    from BiLSTM import BiLSTM
    from SelfAttention import SelfAttention
#    from Dataset import Dataset
else:
#    from .BiLSTM import BiLSTM
    from .SelfAttention import SelfAttention
    #from .Dataset import Dataset


class SentencePresentation(object):
    def __init__(self,vocab, wv_dim=100, lstm_size=128, layers=2, dim_a=200, dim_r=50,
                 alpha=1.0, lr=0.001, norm=5.0, drop_out=0.5, classes=2,adagrad_init_acc=0.1,trunc_norm_init_std = 1e-4, use_embedding=True):
        
        self._lstm_size = lstm_size
        self._alpha = alpha
        self._lr = lr
        self._norm = norm
        self._dr = drop_out
        self._adagrad_init_acc = adagrad_init_acc
        self._vsize = vocab.size()
        self._wv_dim = wv_dim
        self._layers = layers
        self._lstm_size = lstm_size
        self._trunc_norm_init_std = trunc_norm_init_std
        self._embeddings =vocab._word_embedding
        self._use_embedding = use_embedding
        self._classes = classes
        self._dim_a= dim_a
        self._dim_r = dim_r
        self._add_placeholders()
        self._build_graph()
        self._global_step = tf.Variable(0, name='global_step', trainable=False)
        self._add_train_op()


    def _add_placeholders(self):

        self._y = tf.placeholder(dtype=tf.int64, shape=[None], name='input-y')
        self._lens = tf.placeholder(dtype=tf.int64, shape=[None], name='input-lens')
        self._x = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input-x')
        self._dropout = tf.placeholder(tf.float64, name='drop-out')
        self._enc_padding_mask = tf.placeholder(tf.float64, [None, None], name='enc_padding_mask')

    def _build_graph(self):
        #self._mask = tf.sequence_mask(self._lens, dtype=tf.float64)
        with tf.name_scope('input'):
            with tf.variable_scope('embedding'):
                if self._use_embedding:
                    embedding = tf.Variable(self._embeddings, name= "embedding")
                else:
                    embedding = tf.get_variable('embedding', [self._vsize, self._wv_dim], dtype=tf.float64, initializer=tf.truncated_normal_initializer(stddev=self._trunc_norm_init_std))
                embedded_x = tf.nn.embedding_lookup(embedding, self._x)

            with tf.name_scope('multi-bilstm-layer'):
                fw_cells = [DropoutWrapper(BasicLSTMCell(self._lstm_size, state_is_tuple=True),
                                       output_keep_prob=self._dropout) for _ in range(self._layers)]
                bw_cells = [DropoutWrapper(BasicLSTMCell(self._lstm_size, state_is_tuple=True),
                                       output_keep_prob=self._dropout) for _ in range(self._layers)]
                self._fw_cell = MultiRNNCell(fw_cells)
                self._bw_cell = MultiRNNCell(bw_cells)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(self._fw_cell, self._bw_cell, embedded_x, sequence_length=self._lens, dtype=tf.float64)
            output = tf.concat(outputs, axis=2)
            self._output = output
        self._attention = SelfAttention(self._output,self._enc_padding_mask,2*self._lstm_size, dim_a=self._dim_a, dim_r=self._dim_r).get_attention()
        self._attention *= self._enc_padding_mask
        masked_sums = tf.reduce_sum(self._attention, axis=1) # shape (batch_size)
        self._attention /= tf.reshape(masked_sums, [-1, 1]) # re-normalize
        seq_len = tf.shape(output)[1]
        attention = tf.reshape(self._attention, [-1,1,seq_len])
        
        with tf.name_scope('sentence-embedding'):
            self._M = tf.matmul(attention, self._output)

        with tf.name_scope('fully-connected-layer'):
            #self._sentence_embedding = tf.reshape(self._M, shape=[-1, 2*self._dim_r*self._lstm_size])
           
            self._sentence_embedding = tf.reshape(self._M, shape=[-1, 2*self._lstm_size])
            self._fc = slim.fully_connected(self._sentence_embedding, self._classes, activation_fn=None)
            self._pre = tf.nn.softmax(self._fc)

        #with tf.name_scope('penalization'):
        #    AA_T = tf.matmul(self._attention, tf.transpose(self._attention, [0, 2, 1]))
        #    cur_batch = tf.shape(self._x)[0]
        #    I = tf.eye(self._dim_r, batch_shape=[cur_batch], dtype=tf.float64)
        #    self._P = tf.square(tf.norm(AA_T - I, axis=[1, 2], ord='fro'))
        
    def _add_train_op(self):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._fc, labels=self._y)
        self._loss = tf.reduce_mean(loss)
        #self._loss = tf.reduce_mean(loss) + tf.reduce_mean(self._alpha*self._P)
        tvars = tf.trainable_variables()
        gradients = tf.gradients(self._loss, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, self._norm)
        optimizer = tf.train.AdagradOptimizer(self._lr, initial_accumulator_value=self._adagrad_init_acc)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self._global_step, name='train_step')


        with tf.name_scope('acc'):
            pre = tf.argmax(self._fc, axis=1)
            acc = tf.equal(pre, self._y)
            self._acc = tf.reduce_mean(tf.cast(acc, tf.float64))

    def fit(self, sess, batch):
        feed = {self._x: batch.enc_batch, self._y: batch.type, self._lens: batch.enc_lens,self._enc_padding_mask: batch.enc_padding_mask,self._dropout: self._dr}
        loss, pre, global_step,acc,attention, _ = sess.run([self._loss, self._pre, self._global_step, self._acc, self._attention,  self._train_op,], feed_dict=feed)
        return loss,pre ,acc, global_step,attention

    def predict(self, sess, batch):
        feed = {self._x: batch.enc_batch, self._dropout: 1.0, self._lens: batch.enc_lens, self._enc_padding_mask: batch.enc_padding_mask}
        labels, attentions, embedding = sess.run([self._pre, self._attention, self._sentence_embedding], feed_dict=feed)
        return labels, attentions, embedding

    def eval(self, sess, batch):
        feed = {self._x: batch.enc_batch, self._y: batch.type, self._lens: batch.enc_lens,self._enc_padding_mask: batch.enc_padding_mask,self._dropout: self._dr}
        loss, acc = sess.run([self._loss,  self._acc], feed_dict=feed)
        return loss,acc


