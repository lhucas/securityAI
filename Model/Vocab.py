import random
import struct
import csv
from tensorflow.core.example import example_pb2
import json
import gzip
import numpy as np

PAD_TOKEN = '[PAD]' 
UNKNOWN_TOKEN = '[UNK]' 
 

class Vocab(object):
  def __init__(self, vocab_file,vec_file, max_size):
    self._word_to_id = {}
    self._id_to_word = {}
    self._count = 0 
    self._embeddings = {}
    self._word_embedding = np.random.rand(50000,300)

    for w in [UNKNOWN_TOKEN, PAD_TOKEN]:
      self._word_to_id[w] = self._count
      self._id_to_word[self._count] = w
      self._count += 1
    with open(vocab_file, 'r') as vocab_f:
      for line in vocab_f:
          pieces = line.strip().split()
          if len(pieces) != 2:
            print('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
            continue
          w = pieces[1]
          if w in [UNKNOWN_TOKEN, PAD_TOKEN]:
            raise Exception(' [UNK], [PAD] shouldn\'t be in the vocab file, but %s is' % w)
          if w in self._word_to_id.keys():
            raise Exception('Duplicated word in vocabulary file: %s' % w)
          self._word_to_id[w] = self._count
          self._id_to_word[self._count] = w
          self._count += 1
          if max_size != 0 and self._count >= max_size:
            print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count))
            break
      print("Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count-1]))
    
    with open(vec_file, 'r') as pf:
      for line in pf:
          pieces = line.strip().split(" ")
          w = pieces[0]
          self._embeddings[w] = np.array([float(x) for x in pieces[1:]]).astype(np.float32)

    for i in range(len(self._word_to_id)):
      word = self._id_to_word[i]
      if word in self._embeddings.keys():
        self._word_embedding[i] = self._embeddings[word]
  



  def word2id(self, word):
    if word not in self._word_to_id:
      return self._word_to_id[UNKNOWN_TOKEN]
    return self._word_to_id[word]

  def id2word(self, word_id):
    if word_id not in self._id_to_word:
      raise ValueError('Id not found in vocab: %d' % word_id)
    return self._id_to_word[word_id]
	
	
  def decode(self, cur_ids):
        return ' '.join([self.id2word(cur_id) for cur_id in cur_ids])
	
  def example_generator(self,data_path, single_pass, batch_size):
    while True:
      reader = open(data_path, 'r',encoding = "utf-8")
      result = []
      for line in reader:
        result.append(line.strip())
        if len(result)==batch_size:
          yield result
          result = []
      if single_pass:
        print("example_generator completed reading all datafiles. No more data.")
        break

  def example_generator_test(self, data_path, flag):
    print("examplr_generator")
    with open(data_path,"r",encoding="utf-8") as fr:
      info = json.load(fr)
      for i in info:
        context_l = i["contextl"]
        context_ = i["context_"]
        context_r = i["contextr"]
        r_l = []
        r_m = []
        r_r = []
        for c in context_l:
          #print (c)
          r_l.append(c.strip())
        for c in context_r:
          r_r.append(c.strip())
        for c in context_:
          r_m.append(c.strip())
        if flag == "left":
          yield r_l
        elif flag == "right":
          yield r_r
        elif flag == "middle":
          yield r_m


  def example_generator_run(self, data, flag):
    print("examplr_generator")
    for i in data:
      if flag == "left":                                                                                                 
        context = i["contextl"]
      elif flag == "middle":
        context = i["context_"]
      elif flag == "right":
        context = i["contextr"]
      else:
        print("parameter is not right")
      r = []
      for c in context:
        r.append(c.strip())
      yield r


  def size(self):
    return self._count
