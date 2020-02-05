import ujson
import codecs
import random
import os
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn, dynamic_rnn
from layers import multi_conv1d, AttentionCell
from logger import get_logger, Progbar
from tensorflow.contrib.crf import crf_log_likelihood, viterbi_decode
from loss_function import focal_loss
from sklearn.metrics import recall_score, precision_score, f1_score
# from sklearn.metrics import confusion_matrix
#
# def p_r_f1(y_true, y_pred):

def load_data(filename):
    with codecs.open(filename, mode='r', encoding='utf-8') as f:
        dataset = ujson.load(f)
    return dataset


def pad_sequences(sequences, pad=None, max_length=None):
    if pad is None:
        # 0: "PAD" for words and chars, "O" for label
        pad = 0
    if max_length is None:
        max_length = max([len(seq) for seq in sequences])
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq_ = seq[:max_length] + [pad] * max(max_length - len(seq), 0)
        sequence_padded.append(seq_)
        sequence_length.append(min(len(seq), max_length))
    return sequence_padded, sequence_length


def pad_char_sequences(sequences, max_length=None, max_length_2=None):
    sequence_padded, sequence_length = [], []
    if max_length is None:
        max_length = max(map(lambda x: len(x), sequences))
    if max_length_2 is None:
#        print(sequences)
        max_length_2 = max([max(map(lambda x: len(x), seq)) for seq in sequences])
    for seq in sequences:
        sp, sl = pad_sequences(seq, max_length=max_length_2)
        sequence_padded.append(sp)
        sequence_length.append(sl)
    sequence_padded, _ = pad_sequences(sequence_padded, pad=[0] * max_length_2, max_length=max_length)
    sequence_length, _ = pad_sequences(sequence_length, max_length=max_length)
    return sequence_padded, sequence_length


def process_batch_data(batch_words, batch_chars, batch_labels=None):
    b_words, b_words_len = pad_sequences(batch_words)
    b_chars, b_chars_len = pad_char_sequences(batch_chars)
    if batch_labels is None:
        return {"words": b_words, "chars": b_chars, "seq_len": b_words_len, "char_seq_len": b_chars_len,
                "batch_size": len(b_words)}
    else:
        b_labels, _ = pad_sequences(batch_labels)
        return {"words": b_words, "chars": b_chars, "labels": b_labels, "seq_len": b_words_len, "char_seq_len": b_chars_len,
                "batch_size": len(b_words)}


def dataset_batch_iter(dataset, batch_size):
    batch_words, batch_chars, batch_labels = [], [], []
    i = 0
    for record in dataset:
        batch_words.append(record["words"])
        batch_chars.append(record["chars"])
        batch_labels.append(record["labels"])
        if len(record["chars"]) == 0:
            print(i)
        i += 1
        if len(batch_words) == batch_size:
            yield process_batch_data(batch_words, batch_chars, batch_labels)
            batch_words, batch_chars, batch_labels = [], [], []
    if len(batch_words) > 0:
        yield  process_batch_data(batch_words, batch_chars, batch_labels)

def batchnize_dataset(data, batch_size=None, shuffle=True):
    if type(data) == str:
        dataset = load_data(data)
    else:
        dataset = data
    if shuffle:
        random.shuffle(dataset)
    batches = []
    if batch_size is None:
        for batch in dataset_batch_iter(dataset, len(dataset)):
            batches.append(batch)
#        return batches[0]
    else:
        for batch in dataset_batch_iter(dataset, batch_size):
            batches.append(batch)
        return batches


class BiLSTM_model:
    def __init__(self, config, alpha, gamma):
        self.cfg = config
        self.alpha = alpha
        self.gamma = gamma
        # Create folders
        if not os.path.exists(self.cfg["checkpoint_path"]):
            os.makedirs(self.cfg["checkpoint_path"])
        if not os.path.exists(self.cfg["summary_path"]):
            os.makedirs(self.cfg["summary_path"])
        #Create logger
        self.logger = get_logger(os.path.join(self.cfg["checkpoint_path"],str(self.gamma) + str(self.alpha) + "log.txt"))

        # Load dictionary
        dict_data = load_data(self.cfg["vocab"])
        self.word_dict, self.char_dict = dict_data["word_dict"], dict_data["char_dict"]
        self.label_dict = dict_data["label_dict"]
        del dict_data
        self.word_vocab_size = len(self.word_dict)
        self.char_vocab_size = len(self.char_dict)
        self.label_vocab_size = len(self.label_dict)

        self.max_to_keep = self.cfg["max_to_keep"]
        self.checkpoint_path = self.cfg["checkpoint_path"]
        self.summary_path = self.cfg["summary_path"]
        self.word_embedding = self.cfg["word_embedding"]

        self.sess, self.saver = None, None

        # Add placeholder
        self.words = tf.placeholder(tf.int32, shape=[None, None], name="words") # shape = (batch_size, max_time)
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="label") # shape = (batch_size, max_time - 1)
        self.seq_len = tf.placeholder(tf.int32, shape=[None], name="seq_len")
        # shape = (batch_size, max_time, max_word_length)
        self.chars = tf.placeholder(tf.int32, shape=[None, None, None], name="chars")
        self.char_seq_len = tf.placeholder(tf.int32, shape=[None, None], name="char_seq_len")
        # hyper-parameters
        self.is_train = tf.placeholder(tf.bool, shape=[], name="is_train")
        self.batch_size = tf.placeholder(tf.int32, name="batch_size")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_probability")
        self.drop_rate = tf.placeholder(tf.float32, name="dropout_rate")
        self.lr = tf.placeholder(tf.float32, name="learning_rate")

        # Build embedding layer
        with tf.variable_scope("embeddings"):
            self.word_embeddings = tf.Variable(np.load(self.cfg["word_embedding"])["embeddings"], name="embedding",
                                                   dtype=tf.float32, trainable=False)

            word_emb = tf.nn.embedding_lookup(self.word_embeddings, self.words, name="word_emb")
            print("Word embedding shape: {}".format(word_emb.get_shape().as_list()))

            self.char_embeddings = tf.get_variable(name="char_embedding", dtype=tf.float32, trainable=True,
                                                   shape=[self.char_vocab_size, self.cfg["char_emb_dim"]])
            char_emb = tf.nn.embedding_lookup(self.char_embeddings, self.chars, name="chars_emb")
            char_represent = multi_conv1d(char_emb, self.cfg["filter_sizes"], self.cfg["channel_sizes"],
                                          drop_rate=self.drop_rate, is_train=self.is_train)
            print("Chars representation shape: {}".format(char_represent.get_shape().as_list()))
            self.word_emb = tf.concat([word_emb, char_represent], axis=-1)

            self.word_emb = tf.layers.dropout(word_emb, rate=self.drop_rate, training=self.is_train)
            print("Word and chars concatenation shape: {}".format(self.word_emb.get_shape().as_list()))

        # Build model ops
        with tf.name_scope("BiLSTM"):
            with tf.variable_scope('forward'):
                lstm_fw_cell = tf.keras.layers.LSTMCell(self.cfg["num_units"])
            with tf.variable_scope('backward'):
                lstm_bw_cell = tf.keras.layers.LSTMCell(self.cfg["num_units"])
            rnn_outs, *_= bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.word_emb, sequence_length=self.seq_len, dtype=tf.float32)

            # As we have a Bi-LSTM, we have two outputs which are not connected, so we need to merge them.
            rnn_outs = tf.concat(rnn_outs, axis=-1)

#            rnn_outs = tf.layers.dropout(rnn_outs, rate=self.drop_rate, training=self.is_train)
            outputs = rnn_outs
            print("Output shape: {}".format(outputs.get_shape().as_list()))

            self.logits = tf.layers.dense(outputs, units=self.label_vocab_size, use_bias=True)
#            self.logits = tf.nn.softmax(self.logits)
            print("Logits shape: {}".format(self.logits.get_shape().as_list()))
        # Define loss and optimizer
#        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
#        mask = tf.sequence_mask(self.seq_len)
#        self.loss = tf.math.reduce_mean(tf.boolean_mask(losses, mask))
        losses = focal_loss(self.gamma, self.alpha)
        self.loss = losses(self.labels, self.logits)
        self.loss = tf.reduce_mean(self.loss)
        tf.summary.scalar("loss", self.loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)

        print('Params number: {}'.format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)
        self.sess.run(tf.global_variables_initializer())

    def restore_last_session(self, ckpt_path=None):
        if ckpt_path is not None:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
        else:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)  # get checkpoint state
        if ckpt and ckpt.model_checkpoint_path:  # restore session
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def save_session(self, epoch):
        self.saver.save(self.sess, self.checkpoint_path + self.cfg["model_name"], global_step=epoch)

    def close_session(self):
        self.sess.close()

    def _add_summary(self):
        self.summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.summary_path + "train", self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.summary_path + "test")

    def _get_feed_dict(self, batch, keep_prob=0.5, is_train=False, lr=None):
        feed_dict = {self.words: batch["words"], self.seq_len: batch["seq_len"], self.batch_size: batch["batch_size"]}
        if "labels" in batch:
            feed_dict[self.labels] = batch["labels"]
        feed_dict[self.chars] = batch["chars"]
        feed_dict[self.char_seq_len] = batch["char_seq_len"]
        feed_dict[self.keep_prob] = keep_prob
        feed_dict[self.drop_rate] = 1.0 - keep_prob
        feed_dict[self.is_train] = is_train
        if lr is not None:
            feed_dict[self.lr] = lr
        return feed_dict

    def _predict_op(self, data):
        feed_dict = self._get_feed_dict(data)
        pred_logits = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)
        logits = self.sess.run(pred_logits, feed_dict=feed_dict)
        return logits

    def train_epoch(self, train_set,valid_set, epoch):
        num_batches = len(train_set)
        prog = Progbar(target=num_batches)
        for i, batch_data in enumerate(train_set):
            feed_dict = self._get_feed_dict(batch_data, is_train=True, keep_prob=self.cfg["keep_prob"],
                                            lr=self.cfg["lr"])
            _, train_loss, summary = self.sess.run([self.train_op, self.loss, self.summary], feed_dict=feed_dict)
            cur_step = (epoch - 1) * num_batches + (i + 1)
            prog.update(i + 1, [("Global Step", int(cur_step)), ("Train Loss", train_loss)])
            if i % 100 == 0:
                self.train_writer.add_summary(summary, cur_step)
                step = cur_step
        for j, batch_data in enumerate(valid_set):
            feed_dict = self._get_feed_dict(batch_data)
            val_summary = self.sess.run(self.summary, feed_dict=feed_dict)
        self.test_writer.add_summary(val_summary, step)
        micro_f_val, out_str, micro = self.evaluate_punct(valid_set, "val")
        return micro_f_val, train_loss
    def train(self, train_set, valid_set):
        self.logger.info("Start training...")
        best_f1, no_imprv_epoch = -np.inf, 0
        self._add_summary()
        for epoch in range(1, self.cfg["epochs"] + 1):
            self.logger.info('Epoch {}/{}:'.format(epoch, self.cfg["epochs"],))
            micro_f_val, train_loss = self.train_epoch(train_set,valid_set, epoch)  # train epochs
            self.logger.info('Train loss: {} - Valid micro average fscore: {}'.format(train_loss, micro_f_val))
            cur_f1 = micro_f_val
            if cur_f1 > best_f1:
               no_imprv_epoch = 0
               best_f1 = cur_f1
#               f1_test, out_str = self.evaluate_punct(test_set, "test")
#               self.logger.info("\nEvaluate on {} dataset:\n{}\n".format("test", out_str))
               self.save_session(epoch)
            else:
              no_imprv_epoch += 1
              if no_imprv_epoch >= self.cfg["no_imprv_tolerance"]:
                self.logger.info("Early Stopping at epoch - Valid micro average fscore: {:04.2f} - {:04.2f}".format(epoch, best_f1))
                break
        self.train_writer.close()
        self.test_writer.close()
    def test(self,test_set):
        self.logger.info("Start testing...")
        micro_f, out_str, micro = self.evaluate_punct(test_set, "test")
        self.logger.info("\nEvaluate on {} dataset:\n{}\n".format("test", out_str))
        self.logger.info("\n{}\n".format(micro))

    def evaluate_punct(self, dataset, name):
        PUNCTUATIONS = ['O','PERIOD', 'COMMA', 'EXCLAM', 'COLON', 'QMARK','SEMICOLON']
        preds = []
        labels = []

        TP = 0.0
        FP = 0.0
        FN = 0.0

        num_class = len(PUNCTUATIONS)

#        cfm = [ [0 for i in range(7)] for j in range(num_class)]

        for data in dataset:
            predicts = self._predict_op(data)
            for pred, tag, seq_len in zip(predicts, data["labels"], data["seq_len"]):
                preds.append(pred[:seq_len])
#                print(preds)
                labels.append(tag[:seq_len])
                for i in range(len(pred)):
                    for l in range(1,7):
                        if (pred[i] == tag[i]) and (tag[i] == l):
                            TP += 1
                        elif (pred[i] != tag[i]) and (tag[i] == l):
                            FN += 1
                        elif (pred[i] != tag[i]) and (pred[i] == l):
                            FP += 1

        labels = [y for x in labels for y in x]
        preds = [y for x in preds for y in x]

        # metrics = [y for x in labels for y in x]

        precision = precision_score(labels, preds, average=None)
        recall = recall_score(labels, preds, average=None)
        f_score = f1_score(labels, preds, average=None)

        if (TP + FN) != 0:
            micro_r = TP / (TP + FN)
        else:
            micro_r = 0
        ###################
        if (TP + FP) != 0:
            micro_p = TP / (TP + FP)
        else:
            micro_p = 0
        ################
        if (micro_r + micro_p) > 0:
            micro_f = 2*micro_r * micro_p / (micro_r + micro_p)
        else:
            micro_f = 0.0

        micro = 'MICRO AVERAGE:\n\t Precision: ' + str(100*micro_p) + '%\n\tRecall: ' + str(100*micro_r) + ' %\n\t F_1 score: '  + str(100*micro_f) + ' %\n'

        out_str = "-" * 46 + "\n"
        out_str += "{:<16} {:<9} {:<9} {:<9}\n".format("PUNCTUATION", "PRECISION", "RECALL", "F-SCORE")

        for i in range(1,num_class):
            out_str += u"{:<16} {:<9} {:<9} {:<9}\n".format(PUNCTUATIONS[i], "{:.4f}".format(100*precision[i]),
                                                            "{:.4f}".format(100*recall[i]),
                                                            "{:.4f}".format(100*f_score[i]))
        return micro_f, out_str, micro

class BiLSTM_Attention_model:
    def __init__(self, config, alpha, gamma):
        self.cfg = config
        self.alpha = alpha
        self.gamma = gamma
        # Create folders
        if not os.path.exists(self.cfg["checkpoint_path"]):
            os.makedirs(self.cfg["checkpoint_path"])
        if not os.path.exists(self.cfg["summary_path"]):
            os.makedirs(self.cfg["summary_path"])

        #Create logger
        self.logger = get_logger(os.path.join(self.cfg["checkpoint_path"], str(self.gamma) + str(self.alpha) + "log.txt"))

        # Load dictionary
        dict_data = load_data(self.cfg["vocab"])
        self.word_dict, self.char_dict = dict_data["word_dict"], dict_data["char_dict"]
        self.label_dict = dict_data["label_dict"]
        del dict_data
        self.word_vocab_size = len(self.word_dict)
        self.char_vocab_size = len(self.char_dict)
        self.label_vocab_size = len(self.label_dict)

        self.max_to_keep = self.cfg["max_to_keep"]
        self.checkpoint_path = self.cfg["checkpoint_path"]
        self.summary_path = self.cfg["summary_path"]
        self.word_embedding = self.cfg["word_embedding"]

        self.sess, self.saver = None, None

        # Add placeholder
        self.words = tf.placeholder(tf.int32, shape=[None, None], name="words") # shape = (batch_size, max_time)
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="label") # shape = (batch_size, max_time - 1)
        self.seq_len = tf.placeholder(tf.int32, shape=[None], name="seq_len")
        # shape = (batch_size, max_time, max_word_length)
        self.chars = tf.placeholder(tf.int32, shape=[None, None, None], name="chars")
        self.char_seq_len = tf.placeholder(tf.int32, shape=[None, None], name="char_seq_len")
        # hyper-parameters
        self.is_train = tf.placeholder(tf.bool, shape=[], name="is_train")
        self.batch_size = tf.placeholder(tf.int32, name="batch_size")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_probability")
        self.drop_rate = tf.placeholder(tf.float32, name="dropout_rate")
        self.lr = tf.placeholder(tf.float32, name="learning_rate")

        # Build embedding layer
        with tf.variable_scope("embeddings"):
            self.word_embeddings = tf.Variable(np.load(self.cfg["word_embedding"])["embeddings"], name="embedding",
                                                   dtype=tf.float32, trainable=False)

            word_emb = tf.nn.embedding_lookup(self.word_embeddings, self.words, name="word_emb")
            print("Word embedding shape: {}".format(word_emb.get_shape().as_list()))

            self.char_embeddings = tf.get_variable(name="char_embedding", dtype=tf.float32, trainable=True,
                                                   shape=[self.char_vocab_size, self.cfg["char_emb_dim"]])
            char_emb = tf.nn.embedding_lookup(self.char_embeddings, self.chars, name="chars_emb")
            char_represent = multi_conv1d(char_emb, self.cfg["filter_sizes"], self.cfg["channel_sizes"],
                                          drop_rate=self.drop_rate, is_train=self.is_train)
            print("Chars representation shape: {}".format(char_represent.get_shape().as_list()))
            word_emb = tf.concat([word_emb, char_represent], axis=-1)

            self.word_emb = tf.layers.dropout(word_emb, rate=self.drop_rate, training=self.is_train)
            print("Word and chars concatenation shape: {}".format(self.word_emb.get_shape().as_list()))

        # Build model ops
        with tf.name_scope("BiLSTM"):
            with tf.variable_scope('forward'):
                lstm_fw_cell = tf.keras.layers.LSTMCell(self.cfg["num_units"])
            with tf.variable_scope('backward'):
                lstm_bw_cell = tf.keras.layers.LSTMCell(self.cfg["num_units"])
            rnn_outs, *_ = bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.word_emb, sequence_length=self.seq_len,
                                                         dtype=tf.float32)

            # As we have a Bi-LSTM, we have two outputs which are not connected, so we need to merge them.
            rnn_outs = tf.concat(rnn_outs, axis=-1)

            rnn_outs = tf.layers.dropout(rnn_outs, rate=self.drop_rate, training=self.is_train)
            outputs = rnn_outs
            print("Output shape: {}".format(outputs.get_shape().as_list()))
            context = tf.transpose(outputs, [1, 0, 2])
            p_context = tf.layers.dense(outputs, units=2 * self.cfg["num_units"], use_bias=False)
            p_context = tf.transpose(p_context, [1, 0, 2])
            attn_cell = AttentionCell(self.cfg["num_units"], context, p_context)  # time major based
            attn_outs, _ = dynamic_rnn(attn_cell, context, sequence_length=self.seq_len, time_major=True,
                                       dtype=tf.float32)
            outputs = tf.transpose(attn_outs, [1, 0, 2])
            print("Attention output shape: {}".format(outputs.get_shape().as_list()))
            self.logits = tf.layers.dense(outputs, units=self.label_vocab_size, use_bias=True)
#            self.logits = tf.nn.softmax(self.logits)
            print("Logits shape: {}".format(self.logits.get_shape().as_list()))
        # Define loss and optimizer
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        mask = tf.sequence_mask(self.seq_len)
        self.loss = tf.reduce_mean(tf.boolean_mask(losses, mask))
#        losses = focal_loss(self.gamma,self.alpha)
#        self.loss = losses(self.labels, self.logits)
#        self.loss = tf.reduce_mean(self.loss)
        tf.summary.scalar("loss", self.loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)

        print('Params number: {}'.format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)
        self.sess.run(tf.global_variables_initializer())

    def restore_last_session(self, ckpt_path=None):
        if ckpt_path is not None:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
        else:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)  # get checkpoint state
        if ckpt and ckpt.model_checkpoint_path:  # restore session
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def save_session(self, epoch):
        self.saver.save(self.sess, self.checkpoint_path + self.cfg["model_name"], global_step=epoch)

    def close_session(self):
        self.sess.close()

    def _add_summary(self):
        self.summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.summary_path + "train", self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.summary_path + "test")

    def _get_feed_dict(self, batch, keep_prob=1.0, is_train=False, lr=None):
        feed_dict = {self.words: batch["words"], self.seq_len: batch["seq_len"], self.batch_size: batch["batch_size"]}
        if "labels" in batch:
            feed_dict[self.labels] = batch["labels"]
        feed_dict[self.chars] = batch["chars"]
        feed_dict[self.char_seq_len] = batch["char_seq_len"]
        feed_dict[self.keep_prob] = keep_prob
        feed_dict[self.drop_rate] = 1.0 - keep_prob
        feed_dict[self.is_train] = is_train
        if lr is not None:
            feed_dict[self.lr] = lr
        return feed_dict

    def _predict_op(self, data):
        feed_dict = self._get_feed_dict(data)
        pred_logits = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)
        logits = self.sess.run(pred_logits, feed_dict=feed_dict)
        return logits
    def train_epoch(self, train_set,valid_set, epoch):
        num_batches = len(train_set)
        prog = Progbar(target=num_batches)
        for i, batch_data in enumerate(train_set):
            feed_dict = self._get_feed_dict(batch_data, is_train=True, keep_prob=self.cfg["keep_prob"],
                                            lr=self.cfg["lr"])
            _, train_loss, summary = self.sess.run([self.train_op, self.loss, self.summary], feed_dict=feed_dict)
            cur_step = (epoch - 1) * num_batches + (i + 1)
            prog.update(i + 1, [("Global Step", int(cur_step)), ("Train Loss", train_loss)])
            if i % 100 == 0:
                self.train_writer.add_summary(summary, cur_step)
                step = cur_step
        for j, batch_data in enumerate(valid_set):
            feed_dict = self._get_feed_dict(batch_data)
            val_summary = self.sess.run(self.summary, feed_dict=feed_dict)
        self.test_writer.add_summary(val_summary, step)
        micro_f_val, out_str, micro = self.evaluate_punct(valid_set, "val")
        return micro_f_val, train_loss
    def train(self, train_set, valid_set):
        self.logger.info("Start training...")
        best_f1, no_imprv_epoch = -np.inf, 0
        self._add_summary()
        for epoch in range(1, self.cfg["epochs"] + 1):
            self.logger.info('Epoch {}/{}: '.format(epoch, self.cfg["epochs"],))
            micro_f_val, train_loss = self.train_epoch(train_set,valid_set, epoch)  # train epochs
            self.logger.info('Train loss: {} - Valid micro average fscore: {}'.format(train_loss, micro_f_val))
            cur_f1 = micro_f_val
            if cur_f1 > best_f1:
               no_imprv_epoch = 0
               best_f1 = cur_f1
#               f1_test, out_str = self.evaluate_punct(test_set, "test")
#               self.logger.info("\nEvaluate on {} dataset:\n{}\n".format("test", out_str))
               self.save_session(epoch)
            else:
              no_imprv_epoch += 1
              if no_imprv_epoch >= self.cfg["no_imprv_tolerance"]:
                self.logger.info("Early Stopping at epoch - Valid micro average fscore: {:04.2f} - {:04.2f}".format(epoch, best_f1))
                break
        self.train_writer.close()
        self.test_writer.close()
    def test(self,test_set):
        self.logger.info("Start testing...")
        micro_f, out_str, micro = self.evaluate_punct(test_set, "test")
        self.logger.info("\nEvaluate on {} dataset:\n{}\n".format("test", out_str))
        self.logger.info("\n{}\n".format(micro))

    def evaluate_punct(self, dataset, name):
        PUNCTUATIONS = ['O','PERIOD', 'COMMA', 'EXCLAM', 'COLON', 'QMARK','SEMICOLON']
        preds = []
        labels = []

        TP = 0.0
        FP = 0.0
        FN = 0.0

        num_class = len(PUNCTUATIONS)

#        cfm = [ [0 for i in range(7)] for j in range(num_class)]

        for data in dataset:
            predicts = self._predict_op(data)
            for pred, tag, seq_len in zip(predicts, data["labels"], data["seq_len"]):
                preds.append(pred[:seq_len])
#                print(preds)
                labels.append(tag[:seq_len])
                for i in range(len(pred)):
                    for l in range(1,7):
                        if (pred[i] == tag[i]) and (tag[i] == l):
                            TP += 1
                        elif (pred[i] != tag[i]) and (tag[i] == l):
                            FN += 1
                        elif (pred[i] != tag[i]) and (pred[i] == l):
                            FP += 1

        labels = [y for x in labels for y in x]
        preds = [y for x in preds for y in x]

        precision = precision_score(labels, preds, average=None)
        recall = recall_score(labels, preds, average=None)
        f_score = f1_score(labels, preds, average=None)

        if (TP + FN) != 0:
            micro_r = TP / (TP + FN)
        else:
            micro_r = 0
        ###################
        if (TP + FP) != 0:
            micro_p = TP / (TP + FP)
        else:
            micro_p = 0
        ################
        if (micro_r + micro_p) > 0:
            micro_f = 2*micro_r * micro_p / (micro_r + micro_p)
        else:
            micro_f = 0.0

        micro = 'MICRO AVERAGE:\n\t Precision: ' + str(100*micro_p) + '%\n\tRecall: ' + str(100*micro_r) + ' %\n\t F_1 score: '  + str(100*micro_f) + ' %\n'

        out_str = "-" * 46 + "\n"
        out_str += "{:<16} {:<9} {:<9} {:<9}\n".format("PUNCTUATION", "PRECISION", "RECALL", "F-SCORE")

        for i in range(1,num_class):
            out_str += u"{:<16} {:<9} {:<9} {:<9}\n".format(PUNCTUATIONS[i], "{:.4f}".format(100*precision[i]),
                                                            "{:.4f}".format(100*recall[i]),
                                                            "{:.4f}".format(100*f_score[i]))
        return micro_f, out_str, micro

class BiLSTM_CRF_model:
    def __init__(self, config, alpha, gamma):
        self.cfg = config
        self.alpha = alpha
        self.gamma = gamma
        # Create folders
        if not os.path.exists(self.cfg["checkpoint_path"]):
            os.makedirs(self.cfg["checkpoint_path"])
        if not os.path.exists(self.cfg["summary_path"]):
            os.makedirs(self.cfg["summary_path"])

        #Create logger
        self.logger = get_logger(os.path.join(self.cfg["checkpoint_path"], str(self.gamma) + str(self.alpha) + "log.txt"))

        # Load dictionary
        dict_data = load_data(self.cfg["vocab"])
        self.word_dict, self.char_dict = dict_data["word_dict"], dict_data["char_dict"]
        self.label_dict = dict_data["label_dict"]
        del dict_data
        self.word_vocab_size = len(self.word_dict)
        self.char_vocab_size = len(self.char_dict)
        self.label_vocab_size = len(self.label_dict)

        self.max_to_keep = self.cfg["max_to_keep"]
        self.checkpoint_path = self.cfg["checkpoint_path"]
        self.summary_path = self.cfg["summary_path"]
        self.word_embedding = self.cfg["word_embedding"]

        self.sess, self.saver = None, None

        # Add placeholder
        self.words = tf.placeholder(tf.int32, shape=[None, None], name="words") # shape = (batch_size, max_time)
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="label") # shape = (batch_size, max_time)
        self.seq_len = tf.placeholder(tf.int32, shape=[None], name="seq_len")
        # shape = (batch_size, max_time, max_word_length)
        self.chars = tf.placeholder(tf.int32, shape=[None, None, None], name="chars")
        self.char_seq_len = tf.placeholder(tf.int32, shape=[None, None], name="char_seq_len")
        # hyper-parameters
        self.is_train = tf.placeholder(tf.bool, shape=[], name="is_train")
        self.batch_size = tf.placeholder(tf.int32, name="batch_size")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_probability")
        self.drop_rate = tf.placeholder(tf.float32, name="dropout_rate")
        self.lr = tf.placeholder(tf.float32, name="learning_rate")

        # Build embedding layer
        with tf.variable_scope("embeddings"):
            self.word_embeddings = tf.Variable(np.load(self.cfg["word_embedding"])["embeddings"], name="embedding",
                                                   dtype=tf.float32, trainable=False)

            word_emb = tf.nn.embedding_lookup(self.word_embeddings, self.words, name="word_emb")
            print("Word embedding shape: {}".format(word_emb.get_shape().as_list()))

            self.char_embeddings = tf.get_variable(name="char_embedding", dtype=tf.float32, trainable=True,
                                                   shape=[self.char_vocab_size, self.cfg["char_emb_dim"]])
            char_emb = tf.nn.embedding_lookup(self.char_embeddings, self.chars, name="chars_emb")
            char_represent = multi_conv1d(char_emb, self.cfg["filter_sizes"], self.cfg["channel_sizes"],
                                          drop_rate=self.drop_rate, is_train=self.is_train)
            print("Chars representation shape: {}".format(char_represent.get_shape().as_list()))
            word_emb = tf.concat([word_emb, char_represent], axis=-1)

            self.word_emb = tf.layers.dropout(word_emb, rate=self.drop_rate, training=self.is_train)
            print("Word and chars concatenation shape: {}".format(self.word_emb.get_shape().as_list()))

        # Build model ops
        with tf.name_scope("BiLSTM"):
            with tf.variable_scope('forward'):
                lstm_fw_cell = tf.keras.layers.LSTMCell(self.cfg["num_units"])
            with tf.variable_scope('backward'):
                lstm_bw_cell = tf.keras.layers.LSTMCell(self.cfg["num_units"])
            rnn_outs, *_ = bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.word_emb, sequence_length=self.seq_len,
                                                         dtype=tf.float32)

            # As we have a Bi-LSTM, we have two outputs which are not connected, so we need to merge them.
            rnn_outs = tf.concat(rnn_outs, axis=-1)

#            rnn_outs = tf.layers.dropout(rnn_outs, rate=self.drop_rate, training=self.is_train)
            outputs = rnn_outs
            print("Output shape: {}".format(outputs.get_shape().as_list()))

            self.logits = tf.layers.dense(outputs, units=self.label_vocab_size, use_bias=True)
#            self.logits = tf.nn.softmax(self.logits)
            print("Logits shape: {}".format(self.logits.get_shape().as_list()))
        # Define loss and optimizer
        crf_loss, self.trans_params = crf_log_likelihood(self.logits, self.labels, self.seq_len)
        losses = focal_loss(self.gamma,self.alpha)
        self.loss = losses(self.labels, self.logits)
        self.loss = tf.reduce_mean(self.loss)
        tf.summary.scalar("loss", self.loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)

        print('Params number: {}'.format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)
        self.sess.run(tf.global_variables_initializer())


    def restore_last_session(self, ckpt_path=None):
        if ckpt_path is not None:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
        else:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)  # get checkpoint state
        if ckpt and ckpt.model_checkpoint_path:  # restore session
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def save_session(self, epoch):
        self.saver.save(self.sess, self.checkpoint_path + self.cfg["model_name"], global_step=epoch)

    def close_session(self):
        self.sess.close()

    def _add_summary(self):
        self.summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.summary_path + "train", self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.summary_path + "test")

    def _get_feed_dict(self, batch, keep_prob=1.0, is_train=False, lr=None):
        feed_dict = {self.words: batch["words"], self.seq_len: batch["seq_len"], self.batch_size: batch["batch_size"]}
        if "labels" in batch:
            feed_dict[self.labels] = batch["labels"]
        feed_dict[self.chars] = batch["chars"]
        feed_dict[self.char_seq_len] = batch["char_seq_len"]
        feed_dict[self.keep_prob] = keep_prob
        feed_dict[self.drop_rate] = 1.0 - keep_prob
        feed_dict[self.is_train] = is_train
        if lr is not None:
            feed_dict[self.lr] = lr
        return feed_dict

    @staticmethod
    def viterbi_decode(logits, trans_params, seq_len):
        viterbi_sequences = []
        for logit, lens in zip(logits, seq_len):
            logit = logit[:lens]  # keep only the valid steps
            viterbi_seq, viterbi_score = viterbi_decode(logit, trans_params)
            viterbi_sequences += [viterbi_seq]
        return viterbi_sequences

    def _predict_op(self, data):
        feed_dict = self._get_feed_dict(data)
        logits, trans_params, seq_len = self.sess.run([self.logits, self.trans_params, self.seq_len], feed_dict=feed_dict)
        return self.viterbi_decode(logits, trans_params, seq_len)

    def train_epoch(self, train_set,valid_set, epoch):
        num_batches = len(train_set)
        prog = Progbar(target=num_batches)
        for i, batch_data in enumerate(train_set):
            feed_dict = self._get_feed_dict(batch_data, is_train=True, keep_prob=self.cfg["keep_prob"],
                                            lr=self.cfg["lr"])
            _, train_loss, summary = self.sess.run([self.train_op, self.loss, self.summary], feed_dict=feed_dict)
            cur_step = (epoch - 1) * num_batches + (i + 1)
            prog.update(i + 1, [("Global Step", int(cur_step)), ("Train Loss", train_loss)])
            if i % 100 == 0:
                self.train_writer.add_summary(summary, cur_step)
                step = cur_step
        for j, batch_data in enumerate(valid_set):
            feed_dict = self._get_feed_dict(batch_data)
            val_summary = self.sess.run(self.summary, feed_dict=feed_dict)
        self.test_writer.add_summary(val_summary, step)
        micro_f_val, out_str, micro = self.evaluate_punct(valid_set, "val")
        return micro_f_val, train_loss
    def train(self, train_set, valid_set):
        self.logger.info("Start training...")
        best_f1, no_imprv_epoch = -np.inf, 0
        self._add_summary()
        for epoch in range(1, self.cfg["epochs"] + 1):
            self.logger.info('Epoch {}/{}: '.format(epoch, self.cfg["epochs"],))
            micro_f_val, train_loss = self.train_epoch(train_set,valid_set, epoch)  # train epochs
            self.logger.info('Train loss: {} - Valid micro average fscore: {}'.format(train_loss, micro_f_val))
            cur_f1 = micro_f_val
            if cur_f1 > best_f1:
               no_imprv_epoch = 0
               best_f1 = cur_f1
#               f1_test, out_str = self.evaluate_punct(test_set, "test")
#               self.logger.info("\nEvaluate on {} dataset:\n{}\n".format("test", out_str))
               self.save_session(epoch)
            else:
              no_imprv_epoch += 1
              if no_imprv_epoch >= self.cfg["no_imprv_tolerance"]:
                self.logger.info("Early Stopping at epoch - Valid micro average fscore: {:04.2f} - {:04.2f}".format(epoch, best_f1))
                break
        self.train_writer.close()
        self.test_writer.close()
    def test(self,test_set):
        self.logger.info("Start testing...")
        micro_f, out_str, micro = self.evaluate_punct(test_set, "test")
        self.logger.info("\nEvaluate on {} dataset:\n{}\n".format("test", out_str))
        self.logger.info("\n{}\n".format(micro))

    def evaluate_punct(self, dataset, name):
        PUNCTUATIONS = ['O','PERIOD', 'COMMA', 'EXCLAM', 'COLON', 'QMARK','SEMICOLON']
        preds = []
        labels = []

        TP = 0.0
        FP = 0.0
        FN = 0.0

        num_class = len(PUNCTUATIONS)

#        cfm = [ [0 for i in range(7)] for j in range(num_class)]

        for data in dataset:
            predicts = self._predict_op(data)
            for pred, tag, seq_len in zip(predicts, data["labels"], data["seq_len"]):
                preds.append(pred[:seq_len])
#                print(preds)
                labels.append(tag[:seq_len])
                for i in range(len(pred)):
                    for l in range(1,7):
                        if (pred[i] == tag[i]) and (tag[i] == l):
                            TP += 1
                        elif (pred[i] != tag[i]) and (tag[i] == l):
                            FN += 1
                        elif (pred[i] != tag[i]) and (pred[i] == l):
                            FP += 1

        labels = [y for x in labels for y in x]
        preds = [y for x in preds for y in x]

        precision = precision_score(labels, preds, average=None)
        recall = recall_score(labels, preds, average=None)
        f_score = f1_score(labels, preds, average=None)

        if (TP + FN) != 0:
            micro_r = TP / (TP + FN)
        else:
            micro_r = 0
        ###################
        if (TP + FP) != 0:
            micro_p = TP / (TP + FP)
        else:
            micro_p = 0
        ################
        if (micro_r + micro_p) > 0:
            micro_f = 2*micro_r * micro_p / (micro_r + micro_p)
        else:
            micro_f = 0.0
        
        micro = 'MICRO AVERAGE:\n\t Precision: ' + str(100*micro_p) + '%\n\tRecall: ' + str(100*micro_r) + ' %\n\t F_1 score: '  + str(100*micro_f) + ' %\n'

        out_str = "-" * 46 + "\n"
        out_str += "{:<16} {:<9} {:<9} {:<9}\n".format("PUNCTUATION", "PRECISION", "RECALL", "F-SCORE")

        for i in range(1,num_class):
            out_str += u"{:<16} {:<9} {:<9} {:<9}\n".format(PUNCTUATIONS[i], "{:.4f}".format(100*precision[i]),
                                                            "{:.4f}".format(100*recall[i]),
                                                            "{:.4f}".format(100*f_score[i]))
        return micro_f, out_str, micro