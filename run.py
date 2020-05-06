import re
import numpy as np
import json
import tensorflow as tf
from Model import SentencePresentation
#import numpy as np
#from gensim.models import word2vec
#import json
from Model.Vocab import Vocab
from Model.Batcher import Batcher, BatcherTestRun, BatcherTest
import os
import jieba
import pinyin
import fasttext as fasttext

#from Model.Dataset import Dataset
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')

tf.app.flags.DEFINE_string('eval_data_path', './data/eval_data', 'Path expression to tf.Example datafiles. ')
tf.app.flags.DEFINE_string('test_data_path', './data/result_sentence', 'Path expression to tf.Example datafiles. ')
tf.app.flags.DEFINE_string('train_data_path', './data/train_data', 'Path expression to tf.Example datafiles. ')
tf.app.flags.DEFINE_string('vocab_path', './data/vocab', 'Path expression to text vocabulary file.')

tf.app.flags.DEFINE_boolean('single_pass', False, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')
tf.app.flags.DEFINE_integer('train_batch_size', 128, 'train Size of batch.')
tf.app.flags.DEFINE_integer('test_batch_size', 10, 'test Size of batch.')
tf.app.flags.DEFINE_integer('eval_batch_size', 100, 'eval Size of batch.')
tf.app.flags.DEFINE_integer("epoch_num", 30, "num of epoch")
tf.app.flags.DEFINE_integer("example_num", 56426, "num of example")
tf.app.flags.DEFINE_integer('max_seq_size', 100, 'Size of batch.')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')

tf.app.flags.DEFINE_boolean('use_embedding', 'True', 'use pre_trained embedding')
tf.app.flags.DEFINE_string('vec_path', 'data/result', 'pre_trained word embedding path')
tf.app.flags.DEFINE_string('model_path', 'ModelPath', 'model saved path')
tf.app.flags.DEFINE_string('vec_dim', '300', 'word embedding dims')



def getStrAllAplha(str):
    #return pinyin.get_initial(str, delimiter="").upper()
    return pinyin.get(str, format='strip') 
def getStrFirstAplha(str):
    str=getStrAllAplha(str)
    str=str[0:1]
    return str.upper()



def save_session(sess,model_name,saver):
    if not os.path.exists(FLAGS.model_path):
        os.makedirs(FLAGS.model_path)
    saved_name = FLAGS.model_path + model_name
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

def run_test_references(sentences, model_path='reference_model/mini.ftz'):
    model = fasttext.load_model(model_path)
    #result_org = []
    #with open(DATA_DIR, "r") as fr: 
    #    for line in fr:    
    #        result_org.append(line.strip())
    result_org = sentences 
    p = 0
    fw = open("result_reference5", "w")
    finily_result = []
    x_flag = np.zeros(1000)
    for text in result_org:
        p += 1
        c_arr = jieba.cut(text.strip())
        sentence = " ".join(c_arr)
        result = model.predict(sentence)
        num = 0
        candidate = candidates
        s = sentence.split(" ")
        r_w =[]
        j = 0 
        for word in s:
            result = model.predict(word)
            if result[0][0]=="__label__1" and result[1][0] > 0.5:
                r_w.append(j)
            j += 1
        for b in r_w:
            if s[b] in candidates.keys():
                s[b] = candidate[s[b]][0]
                #s[b] = ""
            elif  getStrAllAplha(s[b]) in candidates.keys() :    
                #s[b] = candidates[getStrAllAplha(s[b])][0]
                s[b] = ""
            else:
                s[b] = ""
                """
                flag_find = False
                for key in candidates.keys():
                    if key in s[b]:
                        if x_flag[num] == 1:
                            s[b] = ""
                        else: 
                            if len(s[b]) > 0:
                                x_flag[num] = 1
                            s[b].replace(key, candidates[key][0])
                        flag_find = True
                        break
                if flag_find !=True:
                    if len(re.findall('[a-zA-Z]+',s[b])):
                        #s[b] = s[b][0]+":"+s[b][1:len(s[b])]
                        s[b] = ""
                    else:
                        s[b] = getStrAllAplha(s[b])  
                        #s[b] = ""
                """  
        sentence = "".join(s)
        result = model.predict(sentence)
        finily_result.append(sentence)
        sen = jieba.cut(sentence)
        text = "".join(sen)
        result = model.predict(text)
        fw.write(str(result) + text+"\n")
        print (str(result) + text)
    epoch = 0
    result_over = 0
    fw_r = open("result_reference6", "w")
    while True:
        data = precess_data_new(finily_result)
        fw1 = open("result_reference","w")
        if  epoch  < 3:
        #if True:
            for f_r in finily_result:
                result = model.predict(f_r)
                fw1.write(str(result) + "	" +  f_r + "\n")
                print(f_r)
            target = json.dumps({'text': finily_result}, ensure_ascii=False)
            with open(OUT_DIR, 'w') as f:
                f.write(target)
        #if epoch > 2:
            break
           
        epoch += 1
        print (epoch)
        num = 0
        result_over = 0
        sentences = finily_result
        finily_result = []
        while num < 1000:
            c = 0
            f_text = sentences[num]
            arr = jieba.cut(f_text)
            sen = " ".join(arr)
            r = model.predict(sen)
            arr = sen.split(" ")
            if r == "__label__0":
                finily_result.append("".join(arr))
                result_over += 1
                num += 1
                continue    
            for  c in range(len(data[num]["contextl"])):    
                w = data[num]["contextl"][c]
                result = model.predict(w)
                if result[0][0] == "__label__1":
                    fw_r.write(str(result)+"	"+w+"\n")      
                    s = w.split(" ")
                    if s[c] in candidates.keys():    
                        #s[c] = candidates[s[c]][0]
                        s[c] = ""
                    elif  getStrAllAplha(s[c]) in candidates.keys() :    
                        #s[c] = candidates[getStrAllAplha(s[c])][0]
                        s[c] = ""
                    else:
                        s[c] = ""
                        """
                        flag_find = False
                        for key in candidates.keys():
                            if key in s[c]:
                                s[c] = ""
                                #s[c].replace(key, candidates[key][0])
                                flag_find = True
                                break
                        if flag_find !=True:
                            if len(re.findall('[a-zA-Z]+',s[c])):
                                #s[c]=s[c][0]+":"+s[c][1:len(s[c])]
                                s[c] = ""
                            else: 
                                #if epoch == 2:
                                s[c] = ""
                                #else:
                                #cc  = getStrAllAplha(s[c])
                                #if s[c] == cc:
                                #    s[c] = ""
                                #else:
                                #    s[c] = cc 
                        """    
                    i = 0
                    ret = s[c]
                    arr[c] = ret
                    for con in  data[num]["contextl"]:
                         if i < c:
                             i += 1
                             continue
                         else: 
                             sts = con.split(" ")
                             sts[c] = ret
                             data[num]["contextl"][i] = " ".join(sts) 
                             i += 1
            finily_result.append("".join(arr))       
            num += 1


def run_test_references_new(sentences, model_path='reference_model/mini.ftz'):
    model = fasttext.load_model(model_path)
    result_org = sentences 
    p = 0
    num = 0
    finily_result =[]
    fw_r = open("result_reference","w")
    while num < 1000:
        c = 0
        f_text = sentences[num]
        r = model.predict(f_text)
        if r == "__label__0":
            finily_result.append(f_text)
            num += 1
            continue 
        f_text = list(f_text)       
        for i in range(len(f_text)):
            result = model.predict("".join(f_text[:i+1]))
            
            print (str(result) + "".join(f_text[:i+1]))
            if result[0][0] == "__label__1":
                fw_r.write(str(result)+"	"+"".join(f_text[:i+1]) + "\n")      
                if f_text[i] in candidates.keys():    
                    f_text[i] = candidates[f_text[i]][0]
                elif  getStrAllAplha(f_text[i]) in candidates.keys() :    
                    f_text[i] = ""
                else:
                    f_text[i] = ""
        finily_result.append("".join(f_text))       
        num += 1
    target = json.dumps({'text': finily_result}, ensure_ascii=False)
    with open(OUT_DIR, 'w') as f:
        f.write(target)
     
       

def run_test_reference_m(data, sentences,  model_path="reference_model/mini.ftz"):
    epoch = 0
    model = fasttext.load_model(model_path) 
    f_result = []
    fw = open("result_reference1", "w")
    flag_over = False
    result_over = 0
    while True:
        if epoch > 2 or result_over == 1000:
            target = json.dumps({'text': f_result}, ensure_ascii=False)
            with open(OUT_DIR, 'w') as f:
                f.write(target)
            break
        epoch += 1
        results_l = []
        results_r = []
        b_l_words = []
        b_r_words = []
        num = 0
        words = []
        result_over = 0
        while num < 1000:
            print (num)
            c = 0
            pre_label = "__label__0"
            pre_n = 0
            for  c in range(len(data[num]["context_"])):
                w = data[num]["context_"][c]
                result = model.predict(w)
                #results.append(result)
                fw.write(str(result) + "	"+ w + "\n")
            num += 1
        break





def run_test_reference(data, sentences, model_path="reference_model/mini.ftz"):
    epoch = 0
    model = fasttext.load_model(model_path) 
    f_result = []
    fw = open("result_reference1", "w")
    flag_over = False
    result_over = 0
    while True:
        if epoch > 2 or result_over == 1000:
            target = json.dumps({'text': f_result}, ensure_ascii=False)
            with open(OUT_DIR, 'w') as f:
                f.write(target)
            break
        epoch += 1
        results_l = []
        results_r = []
        b_l_words = []
        b_r_words = []
        num = 0
        words = []
        result_over = 0
        while num < 1000:
            print (num)
            c = 0
            pre_label = "__label__0"
            pre_n = 0
            for  c in range(len(data[num]["contextl"])):
                w = data[num]["contextl"][c]
                result = model.predict(w)
                fw.write(str(result) + "	"+ w + "\n")
                if result[0][0] == "__label__1" and  result[1][0] > 0.5 :           
                    pre_label = result[0][0]  
                    pre_n = result[1][0]
                    s = w.split(" ")
                    if s[c] in candidates.keys():    
                        s[c] = candidates[s[c]][0]
                        #s[c] = ""
                    else:
                        s[c] = ""
                        """
                        flag_find = False
                        for key in candidates.keys():
                            if key in s[c]:
                                s[c].replace(key, candidates[key][0])
                                flag_find = True
                                break
                        if flag_find !=True:
                            if len(re.findall('[a-zA-Z]+',s[c])):
                                s[c]=s[c][0]+"<"+s[c][1:len(s[c])]
                
                            else:
                                s[c] = getStrAllAplha(s[c])  
                                #s[c] = " "
                        """
                    i = 0
                    ret = s[c]
                    for con in  data[num]["contextl"]:
                         if i < c:
                             i += 1
                             continue
                         else: 
                             sts = con.split(" ")
                             sts[c] = ret
                             data[num]["contextl"][i] = " ".join(sts) 
                             i += 1
                else:
                    pre_label = result[0][0]
                    pre_n = result[1][0]    
            num += 1




def run_test(network,vocab,config,saver,data, sentences,candidates):
    sess = tf.Session(config = config)
    sess.run(tf.global_variables_initializer()) 
    _ = load_ckpt(saver, sess) # load a new checkpoint
    epoch = 0
    f_result = []
    fw = open("result_finily", "w")
    flag_over = False
     
    result_over = 0
    while True:
        if epoch > 4 or result_over == 1000:
            return f_result
            target = json.dumps({'text': f_result}, ensure_ascii=False)
            with open(OUT_DIR, 'w') as f:
                f.write(target)
            break

        epoch += 1
        batcher_l = BatcherTestRun(data, vocab, FLAGS.max_seq_size,"left")
        batcher_r = BatcherTestRun(data, vocab, FLAGS.max_seq_size,"right")
        results_l = []
        results_r = []
        b_l_words = []
        b_r_words = []
        num = 0
        c = 0
        words = []
        result_over = 0
        while num < 1000:
            if len(sentences) !=1000:
                print (len(sentences))
                break
            batch_l = batcher_l.next_batch()
            batch_r = batcher_r.next_batch()
            ss = sentences[num].lower()
            batcher = BatcherTest(ss,vocab,100)
            lable, _,_ =  network.predict(sess,batcher.next_batch())
            fw.write(str(num) + "	" +sentences[num] + str(lable[0][1]) + "\n") 
            if int(np.argmax(np.array(lable[0]))) != 1:
                result_over += 1
                num += 1
                word = set()
                words.append(word)
                continue       
            word = set()
            num += 1
            pre_word = []
            n = 0
            while True:
                n+=1
                labels_l, attentions_l, _  = network.predict(sess, batch_l)
                labels_r, attentions_r, _  = network.predict(sess, batch_r)
                result_l = []
                result_r = []
                j = 0
                b_l = 0        
                b_l_word = []
                for w in batch_l.enc_batch:
                    words_json_l = {}
                    w = [vocab.id2word(int(i)) for i in w ]
                    words_json_l[" ".join(w)] = labels_l[j][1]
                    if labels_l[j][1] - b_l > 0.3  and labels_l[j][1] > 0.5:
                        b_l_word.append(j)
                    b_l = labels_l[j][1]
                    j+=1
                    result_l.append(words_json_l)
                results_l.append(result_l)
                b_l_words.append(b_l_word)
                j = 0
                b_r = 1
                b_r_word = []
                for w in batch_r.enc_batch:
                    words_json_r = {}
                    w = [vocab.id2word(int(i)) for i in w]
                    words_json_r[" ".join(w)] = labels_r[j][1]
                    if b_r - labels_r[j][1] > 0.3 and b_r > 0.5 and j != 0:
                        b_r_word.append(j-1)
                    if j == len(batch_r.enc_batch)-1 and labels_r[j][1] > 0.5:
                        b_r_word.append(j)
                    b_r = labels_r[j][1]
                    j+=1
                    result_r.append(words_json_r)
                results_r.append(result_r)
                b_r_words.append(b_r_word)
                if len(b_l_word)!=0:
                    for l in b_l_word:
                        word.add(l)
                        length = len(batch_l.enc_batch)
                        for i in range(0, length):
                            batch_l.enc_batch[i][l] = 0
                if len(b_r_word)!=0:
                     for r in b_r_word:
                        word.add(r) 
                        length = len(batch_r.enc_batch)
                        k = r
                        for i in range(0, length):
                            batch_r.enc_batch[i][k] = 0
                            k = k - 1
                if len(b_l_word) == 0 and len(b_r_word)==0 or pre_word == word or n > 5: 
                    break
                pre_word = word
            words.append(word)
        
        fw_ad = open("ruma_info"+str(epoch), "w", encoding = "utf-8")
        fw_result = open("result_sentence"+str(epoch), "w", encoding = "utf-8")
        finily_result = []
        fw_l = open("result_l.json"+str(epoch), "w", encoding = "utf-8")
        fw_r = open("result_r.json"+str(epoch), "w", encoding ="utf-8")
        fw_l.write(json.dumps(results_l, ensure_ascii=False, indent=4))
        fw_r.write(json.dumps(results_r, ensure_ascii=False,indent=4))   
        for k in range(len(words)):
            sentence = sentences[k]
            s = sentence.split(" ")
            n = 0
            for b in words[k]:
                fw_ad.write(s[b] + " ") 
                if s[b] in candidates.keys(): 
                    #s[b] = ""
                    s[b] = candidates[s[b]][0]
                elif getStrAllAplha(s[b]) in candidates.keys():
                    #s[b] = candidates[getStrAllAplha(s[b])][0]
                    s[b] = ""
                else:
                    s[b] = ""
                    """
                    flag_find = False
                    for key in candidates.keys():
                        if key in s[b]:
                            #if n > 0:
                            #    s[b] = ""
                            #else:
                                #if len(s[b]) > 1:
                                #    n += 1
                                #s[b].replace(key, candidates[key][0])
                            #s[b].replace(key, candidates[key][0])
                            flag_find = True
                            break
                    if flag_find !=True:
                         if len(re.findall('[a-zA-Z]+',s[b])):
                             #s[b]=s[b][0]+":"+s[b][1:len(s[b])]
                             s[b] = "" 
                         else:
                             #s[b] = getStrAllAplha(s[b])  
                             #s[b] = ""
                             s[b] = ""
                    """
            fw_ad.write("        "+sentence + "\n")
            fw_result.write("".join(s)+ "\n")
            finily_result.append("".join(s))
        f_result = finily_result
        sentences_lower,sentences = get_data(finily_result)
        data = precess_data(sentences_lower)      







DATA_DIR = "/tcdata/benchmark_texts.txt"
#DATA_DIR = "ruma_example"
#DATA_DIR = "result_sentence3"
OUT_DIR = "adversarial.txt"


def get_data(contexts):
    data = []
    data_org = []
    for line in contexts:
        c_arr = jieba.cut(line.strip())
        text = " ".join(c_arr)
        data.append(text.lower())
        data_org .append(text)
    return data, data_org



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
            result.append(t)
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
    fw = open("precess_data","w",encoding="utf-8")
    json.dump(contexts, fw, ensure_ascii = False, indent = 4)
    return contexts

def precess_data_new(data):
    contexts = []
    for line in data:
        text_arr = jieba.cut(line.strip())
        text_line = " ".join(text_arr)
        text_arr = text_line.strip().split(" ")
        r = {}
        r["contextl"] = []
        r["context_"] = []
        r["contextr"] = []
        result = []
        i = 0
        for t in text_arr:
            result.append(t)
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
    fw = open("precess_data","w",encoding="utf-8")
    json.dump(contexts, fw, ensure_ascii = False, indent = 4)
    return contexts




def get_test_data():
    result = []
    result_org = []
    #result_text = []
    jieba.load_userdict("keyword")
    with open(DATA_DIR, "r", encoding = "utf-8") as fr:
        for line in fr:
            #result_text.append(line.strip())
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
                keywords[arr[0].strip()] = []
                keywords[getStrAllAplha(arr[0].strip())] = []
                for i in range(1,len(arr)):
                        keywords[arr[0].strip()].append(arr[i].strip())
                        keywords[getStrAllAplha(arr[0].strip())].append(arr[i].strip())
        print (keywords)
        return keywords
        

if __name__ == "__main__":
    data ,data_org = get_test_data()
    candidates = get_candidate("candidate")
    context = precess_data(data) 
    tf.logging.set_verbosity(tf.logging.INFO)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    vocab = Vocab(FLAGS.vocab_path,FLAGS.vec_path, FLAGS.vocab_size, FLAGS.vec_dim)
    network =SentencePresentation(vocab, wv_dim=100, lstm_size=128, layers=1, dim_a=10,dim_r=30, classes=2,  norm=0.5, lr=0.01,adagrad_init_acc = FLAGS.adagrad_init_acc ,trunc_norm_init_std =FLAGS.trunc_norm_init_std ,use_embedding = FLAGS.use_embedding)


    saver = tf.train.Saver(max_to_keep=3)
    #run_test_references_new(text)
    result = run_test(network,vocab,config,saver,context, data_org, candidates)
    #run_test_references(result)
    #run_test_reference_m(context,data_org)
    #result = run_test_reference(context, data_org)
