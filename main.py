import tensorflow as tf
from Model import SentencePresentation
import numpy as np
#from gensim.models import word2vec
import json
from Model.Vocab import Vocab
from Model.Batcher import Batcher, BatcherTest
import os

#from Model.Dataset import Dataset
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')

tf.app.flags.DEFINE_string('eval_data_path', './data/eval_data', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('test_data_path', './data/result_sentence', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('train_data_path', './data/train_data', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')

tf.app.flags.DEFINE_string('vocab_path', './data/vocab', 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_boolean('single_pass', False, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')
tf.app.flags.DEFINE_integer('train_batch_size', 128, 'Size of batch.')
tf.app.flags.DEFINE_integer('test_batch_size', 10, 'Size of batch.')
tf.app.flags.DEFINE_integer('eval_batch_size', 100, 'Size of batch.')
tf.app.flags.DEFINE_integer("epoch_num", 30, "num of epoch")
tf.app.flags.DEFINE_integer("example_num", 56426, "num of example")
tf.app.flags.DEFINE_integer('max_seq_size', 100, 'Size of batch.')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')

tf.app.flags.DEFINE_boolean('use_embedding', 'True', 'must be one of train/eval/decode')
tf.app.flags.DEFINE_string('vec_path', './data/result', 'must be one of train/eval/decode')
tf.app.flags.DEFINE_string('model_path', 'ModelPath', 'must be one of train/eval/decode')


def save_session(sess,model_name,saver):
    if not os.path.exists(FLAGS.model_path):
        os.makedirs(FLAGS.model_path)
    saved_name = os.path.join(FLAGS.model_path , model_name)
    saver.save(sess, saved_name)

    

def load_ckpt(saver, sess):
  """Load checkpoint from the ckpt_dir (if unspecified, this is train dir) and restore it to saver and sess, waiting 10 secs in the case of failure. Also returns checkpoint name."""
  while True:
    try:
      #latest_filename = "checkpoint_best" if ckpt_dir=="eval" else None
      #ckpt_dir = os.path.join(FLAGS.log_root, ckpt_dir)
      ckpt_state = tf.train.get_checkpoint_state("./ModelPath")
      tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
      saver.restore(sess, ckpt_state.model_checkpoint_path)
      return ckpt_state.model_checkpoint_path
    except:
      tf.logging.info("Failed to load checkpoint from %s. Sleeping for %i secs...", ckpt_dir, 10)
      time.sleep(10)



def run_predict(network,vocab,config,saver):
    sess = tf.Session(config = config)
    sess.run(tf.global_variables_initializer())
    _ = load_ckpt(saver, sess) # load a new checkpoint
    batcher = Batcher(FLAGS.test_data_path, vocab, FLAGS.test_batch_size, True,FLAGS.max_seq_size) 
    results = []
    error = []
    num = 0
    while True:
        batch = batcher.next_batch()
        labels, attentions, embedding = network.predict(sess, batch)
        num += 1
        result = []
        types = batch.type
        j = 0
        enc_lens = batch.enc_lens
        for w, attention in zip(batch.enc_batch, attentions):
            if  int(np.argmax(np.array(labels[j])))!=int(types[j]):
                context = " ".join([vocab.id2word(int(i)) for i in w])
                print (str(j) + str(types[j]) + "  " + context)
                error.append(str(types[j]) + "	" + context )         
            words_json = {}
            w = [vocab.id2word(int(i)) for i in w]
            lens = 0
            for c, a in zip(w, attention):
                words_json[c] = a
                if lens >= enc_lens[j]:
                    break
                lens += 1
            result.append(words_json)
            j += 1
        results.append(result)
        if num>=200:
            break
    fw = open("result.json", "w", encoding = "utf-8")
    fw.write(json.dumps(results,  ensure_ascii=False, indent = 4))
    fw1 = open("errorcases", "w", encoding = "utf-8")
    for e in error:
        print (e)
        fw1.write(e + "\n")

def run_eval(sess, vocab):
    batcher = Batcher(FLAGS.eval_data_path, vocab, FLAGS.eval_batch_size, True, FLAGS.max_seq_size)
    batch = batcher.next_batch()
    loss,acc = network.eval(sess, batch)


def run_test(network,vocab,config,saver):
    sess = tf.Session(config = config)
    sess.run(tf.global_variables_initializer())
    _ = load_ckpt(saver, sess) # load a new checkpoint
    batcher = BatcherTest(FLAGS.test_data_path, vocab, FLAGS.max_seq_size,"middle")
    results = []
    error = []
    rr = []
    num = 0
    while num < 1000:
        batch = batcher.next_batch()
        labels, attentions, embedding  = network.predict(sess, batch)
        num += 1
        result = []
        types = batch.type
        print (num)
        j = 0
        enc_lens = batch.enc_lens
        for w in batch.enc_batch:
            print (j)
            #if  int(np.argmax(np.array(labels[j])))!=int(types[j]):
            #    context = " ".join([vocab.id2word(int(i)) for i in w])
            #    print (str(j) + str(types[j]) + "  " + context)
            #    error.append(str(types[j]) + "  " + context )
            words_json = {}
            w = [vocab.id2word(int(i)) for i in w]
            print (w)
            lens = 0
            #for c, a in zip(w, attention):
                #words_json[c] = a
            #    if lens >= enc_lens[j]:
            #        break
            #    lens += 1
 
            words_json[" ".join(w)] = labels[j][1]
            j+=1
            result.append(words_json)
        results.append(result)
        
    fw = open("result_m.json", "w", encoding = "utf-8")
    fw.write(json.dumps(results,  ensure_ascii=False, indent = 4))
    #fw1 = open("errorcases", "w", encoding = "utf-8")
    #for e in error:
    #    fw1.write(e + "\n")
  


def run_train(network,vocab, config,saver):
     with tf.Session(config = config) as sess:
        sess.run(tf.global_variables_initializer())
        before_loss = 500000
        #saver = tf.train.Saver(max_to_keep=3)
        s = 0
        batcher = Batcher(FLAGS.train_data_path, vocab, FLAGS.train_batch_size, FLAGS.single_pass,FLAGS.max_seq_size)
        while True: 
            batch = batcher.next_batch()
            loss,pre, acc ,steps, attention= network.fit(sess, batch)
            s += 1
            if steps % 100 == 0 and steps!=0:
                print('steps: %d, loss: %f, acc: %f' % (steps, loss, acc))
                bu = 0
                eval_loss = 0
                eval_acc =0
                while True:
                    batcher_eval = Batcher(FLAGS.eval_data_path, vocab, FLAGS.eval_batch_size, True, FLAGS.max_seq_size)
                    batch_eval = batcher_eval.next_batch()
                    acc_loss,acc_acc = network.eval(sess, batch_eval)
                    bu += 100
                    eval_loss += acc_loss
                    eval_acc += acc_acc
                    if bu >=3000:
                        break
                print('acc_loss:%f, acc_acc:%f'%(eval_loss/30, eval_acc/30))                       
                if loss < before_loss:
                    save_session(sess,"early_best"+str(steps) + ".ckpt",saver)
                    before_loss = loss 
            if steps >=  FLAGS.example_num / FLAGS.train_batch_size * FLAGS.epoch_num:
                break
         
if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    print ("read vocab")
    vocab = Vocab(FLAGS.vocab_path,FLAGS.vec_path, FLAGS.vocab_size)
    print ("read over")
    #batcher = Batcher(FLAGS.train_data_path, vocab, FLAGS.batch_size, FLAGS.single_pass,FLAGS.max_seq_size)
    network =SentencePresentation(vocab, wv_dim=100, lstm_size=128, layers=1, dim_a=10,dim_r=30, classes=2,  norm=0.5, lr=0.01,adagrad_init_acc = FLAGS.adagrad_init_acc ,trunc_norm_init_std =FLAGS.trunc_norm_init_std ,use_embedding = FLAGS.use_embedding)
    saver = tf.train.Saver(max_to_keep=3)
    #run_predict(network, vocab, config, saver)
    run_train(network, vocab,config,saver)
    #run_test(network,vocab,config,saver)
