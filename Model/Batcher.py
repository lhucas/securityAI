import queue as Queue
from random import shuffle
from threading import Thread
import time
import numpy as np
import tensorflow as tf
import Model.Vocab

class Example(object):
  def __init__(self, article, types, vocab, max_enc_steps):
    article_words = article.split(" ")
    if len(article_words) > max_enc_steps:
      article_words = article_words[:max_enc_steps]
    self.enc_len = len(article_words) 
    self.enc_input = [vocab.word2id(w) for w in article_words] 
    self.type = types

  def pad_encoder_input(self, max_len, pad_id):
    while len(self.enc_input) < max_len:
      self.enc_input.append(pad_id)



class Batch(object):
  def __init__(self, example_list, batch_size, vocab):
    self.pad_id = vocab.word2id(Model.Vocab.PAD_TOKEN)
    self.vocab = vocab
    self.init_encoder_seq(example_list, batch_size)

  def init_encoder_seq(self,example_list, batch_size):
    max_enc_seq_len = max([ex.enc_len for ex in example_list])
    for ex in example_list:
      ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

    self.enc_batch = np.zeros((batch_size, max_enc_seq_len), dtype=np.int32)
    self.enc_lens = np.zeros((batch_size), dtype=np.int32)
    self.type = np.zeros((batch_size), dtype=np.int32)
    self.enc_padding_mask = np.zeros((batch_size, max_enc_seq_len), dtype=np.float32)
    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
      self.enc_batch[i, :] = ex.enc_input[:]
      self.type[i] = ex.type
      self.enc_lens[i] = ex.enc_len
      for j in range(ex.enc_len):
        self.enc_padding_mask[i][j] = 1



class Batcher(object): 
  #BATCH_QUEUE_MAX = 100 

  def __init__(self, data_path, vocab, batch_size, single_pass, max_seq_len):
    self._data_path = data_path
    self._vocab = vocab
    self._batch_size = batch_size
    self._single_pass = single_pass
    self._max_seq_len = max_seq_len
    self._input_gen = self._vocab.example_generator(self._data_path, self._single_pass, self._batch_size)

  def next_batch(self):
    try:
      content = next(self._input_gen)
      examples = []
      for c in content:
        con_arr = c.split("\t")
        types = int(con_arr[0])
        text = con_arr[1]
        example = Example(text, types, self._vocab,self._max_seq_len)
        examples.append(example)
      batch = Batch(examples, self._batch_size, self._vocab)
      return  batch
    except StopIteration:
      tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
      if self._single_pass:
        tf.logging.info("single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
        self._finished_reading = True
        #break
      else:
        raise Exception("single_pass mode is off but the example generator is out of data; error.")



class BatcherTest(object):
  def __init__(self, data_path, vocab,  max_seq_len, flag):
    self._data_path = data_path
    self._vocab = vocab
    self._max_seq_len = max_seq_len
    self._input_gen = self._vocab.example_generator_test(self._data_path, flag)
    self._flag = flag

  def next_batch(self):
    print ("next_batch!!!!")
    try:
      content = next(self._input_gen)
      examples = []
      for c in content:
        print (c)
        #con_arr = c.split("     ")
        #types = int(con_arr[0])
        #text = con_arr[1]
        example = Example(c, 1, self._vocab,self._max_seq_len)
        examples.append(example)
      batch = Batch(examples, len(content), self._vocab)
      return  batch
    except StopIteration:
      tf.logging.info("The example generator for this example queue filling thread has exhausted data.")





class BatcherTestRun(object):
  def __init__(self, data, vocab,  max_seq_len, flag):
    self._data = data
    self._vocab = vocab
    self._max_seq_len = max_seq_len
    self._input_gen = self._vocab.example_generator_run(self._data, flag)
    self._flag = flag

  def next_batch(self):
    print ("next_batch!!!!")
    try:
      content = next(self._input_gen)
      examples = []
      for c in content:
        print (c)
        example = Example(c, 1, self._vocab,self._max_seq_len)
        examples.append(example)
      batch = Batch(examples, len(content), self._vocab)
      return  batch
    except StopIteration:
      tf.logging.info("The example generator for this example queue filling thread has exhausted data.")



""" 
  def fill_example_queue(self):
    input_gen = self._vocab.example_generator(self._data_path, self._single_pass)
    tf.logging.info(input_gen)
    while True:
      try:
        content = next(input_gen)
        #tf.logging.info(content) 
        #print(content)
        con_arr = content.split("	")
        types = int(con_arr[0])
        text = con_arr[1]
        #tf.logging.info(con_arr[0] + "	"+text)
      except StopIteration:
        tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
        if self._single_pass:
          tf.logging.info("single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
          self._finished_reading = True
          break
        else:
          raise Exception("single_pass mode is off but the example generator is out of data; error.")

     
      example = Example(text, types, self._vocab,self._max_seq_len) 
      self._example_queue.put(example) 


  def fill_batch_queue(self):
    while True:
      inputs = []
      for _ in range(self._batch_size * self._bucketing_cache_size):
        inputs.append(self._example_queue.get())
      inputs = sorted(inputs, key=lambda inp: inp.enc_len) 
      batches = []
      for i in range(0, len(inputs), self._batch_size):
        batches.append(inputs[i:i + self._batch_size])
      if not self._single_pass:
        shuffle(batches)
      for b in batches:
        self._batch_queue.put(Batch(b, self._batch_size, self._vocab))


  def watch_threads(self):
    while True:
      time.sleep(60)
      for idx,t in enumerate(self._example_q_threads):
        if not t.is_alive(): # if the thread is dead
          tf.logging.error('Found example queue thread dead. Restarting.')
          new_t = Thread(target=self.fill_example_queue)
          self._example_q_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()
      for idx,t in enumerate(self._batch_q_threads):
        if not t.is_alive(): 
          tf.logging.error('Found batch queue thread dead. Restarting.')
          new_t = Thread(target=self.fill_batch_queue)
          self._batch_q_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()
"""
