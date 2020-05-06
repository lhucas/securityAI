import re
#import json
import tensorflow as tf
from Model import SentencePresentation
#import numpy as np
#from gensim.models import word2vec
#import json
from Model.Vocab import Vocab
from Model.Batcher import Batcher, BatcherTestRun
import os
import jieba
import pinyin


#from Model.Dataset import Dataset
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')

tf.app.flags.DEFINE_string('eval_data_path', 'data/eval_data', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('test_data_path', 'data/test_data', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('train_data_path', 'data/train_data', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')

tf.app.flags.DEFINE_string('vocab_path', 'data/vocab', 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_boolean('single_pass', False, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')
tf.app.flags.DEFINE_integer('test_batch_size', 1, 'Size of batch.')
tf.app.flags.DEFINE_integer('max_seq_size', 100, 'Size of batch.')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')
tf.app.flags.DEFINE_string('model_path', 'ModelPath', 'must be one of train/eval/decode')



def getStrAllAplha(str):
    return pinyin.get_initial(str, delimiter="").upper()
    
def getStrFirstAplha(str):
    str=getStrAllAplha(str)
    str=str[0:1]
    return str.upper()



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
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.model_path)
      tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
      saver.restore(sess, ckpt_state.model_checkpoint_path)
      return ckpt_state.model_checkpoint_path
    except:
      tf.logging.info("Failed to load checkpoint from %s. Sleeping for %i secs...", ckpt_dir, 10)
      time.sleep(10)


def run_test(network,vocab,config,saver,data, sentences,candidates):
    sess = tf.Session(config = config)
    sess.run(tf.global_variables_initializer())
    _ = load_ckpt(saver, sess) # load a new checkpoint
    batcher_l = BatcherTestRun(data, vocab, FLAGS.max_seq_size,"left")
    batcher_m = BatcherTestRun(data, vocab, FLAGS.max_seq_size,"middle")
    batcher_r = BatcherTestRun(data, vocab, FLAGS.max_seq_size,"right")
    results_l = []
    results_r = []
    results_m = []
    b_l_words = []
    b_r_words = []
    b_m_words = []
    num = 0
    c = 0
    while num <1000:
        batch_l = batcher_l.next_batch()
        labels_l, attentions_l, _  = network.predict(sess, batch_l)
        batch_r = batcher_r.next_batch()
        labels_r, attentions_r, _  = network.predict(sess, batch_r)
        batch_m = batcher_m.next_batch()
        labels_m, attentions_m, _  = network.predict(sess, batch_m)
        num += 1
        result_l = []
        result_r = []
        result_m = []
        j = 0
        b_l = 0	
        b_l_word = []
        
        for w in batch_l.enc_batch:
            words_json_l = {}
            w = [vocab.id2word(int(i)) for i in w ]
            lens = 0
            words_json_l[" ".join(w)] = labels_l[j][1]
            if labels_l[j][1] - b_l > 0.5:
                b_l_word.append(j)
            b_l = labels_l[j][1]
            j+=1
            result_l.append(words_json_l)
        if labels_l[j-1][1] > 0.5:
            c += 1


        results_l.append(result_l)
        b_l_words.append(b_l_word)
        j = 0
        b_r = 1
        b_r_word = []
        for w in batch_r.enc_batch:
            words_json_r = {}
            w = [vocab.id2word(int(i)) for i in w]
            lens = 0
            words_json_r[" ".join(w)] = labels_r[j][1]
            if b_r - labels_r[j][1] > 0.5 and j != 0:
                b_r_word.append(j-1)
            if j == len(batch_r.enc_batch)-1 and labels_r[j][1] > 0.5:
                b_r_word.append(j)
            b_r = labels_r[j][1]
            j+=1
            result_r.append(words_json_r)
        
        results_r.append(result_r)
        b_r_words.append(b_r_word)
        j = 0       
        b_m_word = []
        for w in batch_m.enc_batch:
            words_json_m = {}
            w = [vocab.id2word(int(i)) for i in w]
            lens = 0
            words_json_m[" ".join(w)] = labels_m[j][1]
            if j==0  and labels_m[j+1][1]-labels_m[j][1]>0.5:
                b_m_word.append(j)
            elif j==(len(batch_m.enc_batch)-1):
                break 
                #b_m_word.append(j)
            elif j!=0 and j!=len(batch_m.enc_batch)-1 and labels_m[j+1][1]-labels_m[j][1] > 0.5 and labels_m[j-1][1]-labels_m[j][1]>0.5:
                b_m_word.append(j)
                
            j+=1
            result_m.append(words_json_m)
        results_m.append(result_m)
        b_m_words.append(b_m_word)
    """
    fw_l = open("result_l.json", "w", encoding = "utf-8")
    fw_l.write(json.dumps(results_l,  ensure_ascii=False, indent = 4))

    fw_r = open("result_r.json", "w", encoding = "utf-8")
    fw_r.write(json.dumps(results_r,  ensure_ascii=False, indent = 4))

    fw_m = open("result_m.json", "w", encoding = "utf-8")
    fw_m.write(json.dumps(results_m,  ensure_ascii=False, indent = 4))
    """
    fw_ad = open("ruma_info", "w", encoding = "utf-8")
    fw_ruma = open("rumaci", "w",encoding = "utf-8")
    fw_result = open("adversarial.txt", "w", encoding = "utf-8")
    r_index = []
    for ind in range(len(b_l_words)):
        rs = set()
        for l in b_l_words[ind]:
            rs.add(l)
        for r in b_r_words[ind]:
            rs.add(r)
        for m in b_m_words[ind]:
            rs.add(m)
        r_index.append(rs)
    rumaci = set()
    for k in range(len(b_l_words)):
        sentence = sentences[k]
        s = sentence.split(" ")
        for b in r_index[k]:
            rumaci.add(s[b]) 
            fw_ad.write(s[b] + " ") 
            print (s[b])
            if len(re.findall('[a-zA-Z]+',s[b])):
                s[b]=s[b][0]+"."+s[b][1:len(s[b])]
                print (s[b])
            elif s[b] in candidates.keys():    
                s[b] = candidates[s[b]]
            else:
                s[b] = getStrAllAplha(s[b])  
        fw_ad.write("	"+sentence + "\n")
        fw_result.write( "".join(s)+ "\n")
    for w in rumaci:
        fw_ruma.write(w + "\n")

#DATA_DIR = "/tcdata/benchmark_texts.txt"
DATA_DIR = "ruma_example"
def precess_data(data):
    contexts = []
    for line in data:
        text_arr = line.strip().split(" ")
        r = {}
        r["contextl"] = []
        r["context_"] = []
        r["contextr"] = []
        result = []
        i = 0
        for t in text_arr:
            result.append(t.strip())
            i += 1
            r["contextl"].append(" ".join(result[:i]))
        l = len(result)
        j = 0
        while j <= len(result):
            if j == len(result):
                r["context_"] .append(" ".join( result[:j]))
            elif j == 0:
                r["context_"].append(" ".join(result[j+1:]))
            else:
                r["context_"] .append(" ".join(result[:j]+result[j+1:]))
            if j != len(result):
                r["contextr"] .append(" ".join(result[j:]))
            j += 1
        contexts.append(r)  
    #fw = open("precess_data","w",encoding="utf-8")
    #json.dump(contexts, fw, ensure_ascii = False, indent = 4)
    return contexts

def get_test_data():
    result = []
    result_org = []
    jieba.load_userdict("keyword")
    with open(DATA_DIR, "r", encoding = "utf-8") as fr:
        for line in fr:
            c_arr = jieba.cut(line.strip())
            rr = " ".join(c_arr)	
            result_org.append(rr)
            rr = rr.lower()
            result.append(rr)
    return result, result_org

def get_candidate(file_r):
	keywords = {}
	fr = open(file_r, "r", encoding="utf-8")
	for line in fr:
		arr = line.strip().split("\t")
		keywords[arr[0].strip()]=arr[1].strip()
	print (keywords)
	return keywords
	

if __name__ == "__main__":
    data ,data_org = get_test_data()
    candidates = get_candidate("candidate")
    context = precess_data(data) 
    tf.logging.set_verbosity(tf.logging.INFO)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)
    network =SentencePresentation(vocab, wv_dim=100, lstm_size=128, layers=1, dim_a=10,dim_r=30, classes=2,  norm=0.5, lr=0.01,adagrad_init_acc = FLAGS.adagrad_init_acc ,trunc_norm_init_std =FLAGS.trunc_norm_init_std )
    saver = tf.train.Saver(max_to_keep=3)
    run_test(network,vocab,config,saver,context, data_org, candidates)
