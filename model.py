import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from data_utils import minibatches, pad_sequences
from general_utils import Progbar, print_sentence
from sklearn.metrics import f1_score


class NERModel(object):
    def __init__(self, config, embeddings, ntags, nchars):
        self.config = config
        self.embeddings = embeddings
        self.nchars = nchars
        self.ntags = ntags
        self.logger = config.logger

        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                                       name="word_ids")

        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                                               name="sequence_lengths")

        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                                       name="char_ids")

        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                                           name="word_lengths")

        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                                     name="labels")

        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                                 name="lr")

        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings, name="_word_embeddings", dtype=tf.float32,
                                           trainable=self.config.train_embeddings)
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids,
                                                     name="word_embeddings")
            print(word_embeddings)

        with tf.variable_scope("chars"):
            _char_embeddings = tf.get_variable(name="_char_embeddings", dtype=tf.float32,
                                               shape=[self.nchars, self.config.dim_char])
            char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.char_ids,
                                                     name="char_embeddings")
            shape = tf.shape(char_embeddings)
            char_embeddings = tf.reshape(char_embeddings, shape=[-1, shape[-2], self.config.dim_char])
            word_lengths = tf.reshape(self.word_lengths, shape=[-1])
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.char_hidden_size,
                                              state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.char_hidden_size,
                                              state_is_tuple=True)

            _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                                  cell_bw,
                                                                                  inputs=char_embeddings,
                                                                                  sequence_length=word_lengths,
                                                                                  dtype=tf.float32)

            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.reshape(output, shape=[-1, shape[1], 2 * self.config.char_hidden_size])

            # word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size)
            # cell_fw = tf.contrib.rnn.MultiRNNCell([cell_fw] * 3, state_is_tuple=True)
            # print(self.word_embeddings)
            # cell_bw = tf.contrib.rnn.MultiRNNCell([cell_bw] * 3, state_is_tuple=True)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                        cell_bw, self.word_embeddings,
                                                                        sequence_length=self.sequence_lengths,
                                                                        dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", shape=[2 * self.config.hidden_size, self.ntags],
                                dtype=tf.float32)

            b = tf.get_variable("b", shape=[self.ntags], dtype=tf.float32,
                                initializer=tf.zeros_initializer())

            ntime_steps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2 * self.config.hidden_size])
            # Highway Layer
            output = self.highway(output, 2 * self.config.hidden_size, tf.nn.relu)
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, ntime_steps, self.ntags])

        log_likelihood, self.transition_params = crf_log_likelihood(
            self.logits, self.labels, self.sequence_lengths)
        self.loss = tf.reduce_mean(-log_likelihood)

        tf.summary.scalar("loss", self.loss)

        with tf.variable_scope("train_step"):
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)

        self.init = tf.global_variables_initializer()

    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        char_ids, word_ids = zip(*words)
        word_ids, sequence_lengths = pad_sequences(word_ids, 0)
        char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)

        feed = {self.word_ids: word_ids, self.sequence_lengths: sequence_lengths, self.char_ids: char_ids,
                self.word_lengths: word_lengths}

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths

    def highway(self, x, size, activation, carry_bias=-1.0):
        W_T = tf.Variable(tf.truncated_normal([size, size], stddev=0.1), name="weight_transform")
        b_T = tf.Variable(tf.constant(carry_bias, shape=[size]), name="bias_transform")
        W = tf.Variable(tf.truncated_normal([size, size], stddev=0.1), name="weight")
        b = tf.Variable(tf.constant(0.1, shape=[size]), name="bias")
        T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name="transform_gate")
        H = activation(tf.matmul(x, W) + b, name="activation")
        C = tf.subtract(1.0, T, name="carry_gate")
        y = tf.add(tf.multiply(H, T), tf.multiply(x, C), "y")
        return y

    def add_summary(self, sess):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.output_path, sess.graph)

    def predict_batch(self, sess, words):
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        viterbi_sequences = []
        logits, transition_params = sess.run([self.logits, self.transition_params],
                                             feed_dict=fd)
        # iterate over the sentences
        for logit, sequence_length in zip(logits, sequence_lengths):
            # keep only the valid time steps
            logit = logit[:sequence_length]
            viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
                logit, transition_params)
            viterbi_sequences += [viterbi_sequence]

        return viterbi_sequences, sequence_lengths

    def run_epoch(self, sess, train, dev, tags, epoch):
        nbatches = (len(train) + self.config.batch_size - 1) // self.config.batch_size
        prog = Progbar(target=nbatches)
        for i, (words, labels) in enumerate(minibatches(train, self.config.batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr, self.config.dropout)

            _, train_loss, summary = sess.run([self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch * nbatches + i)

        acc, f1 = self.evaluate(sess, dev, tags)
        self.logger.info("- dev acc {:04.2f} - f1 {:04.2f}".format(100 * acc, 100 * f1))
        return acc, f1

    def evaluate(self, sess, test, tags):
        accuracy = []
        f1 = []
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(sess, words)
            for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                accuracy += [a == b for (a, b) in zip(lab, lab_pred)]
                f1.append(f1_score(lab, lab_pred, average='macro'))

        acc = np.mean(accuracy)
        f1 = sum(f1) / float(len(f1))
        return acc, f1

    def train(self, train, dev, tags):
        best_score = 0
        saver = tf.train.Saver()
        # for early stopping
        nepoch_no_imprv = 0
        with tf.Session() as sess:
            sess.run(self.init)
            self.add_summary(sess)
            for epoch in range(self.config.nepochs):
                self.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.config.nepochs))
                acc, f1 = self.run_epoch(sess, train, dev, tags, epoch)
                self.config.lr *= self.config.lr_decay

                if f1 >= best_score:
                    nepoch_no_imprv = 0
                    if not os.path.exists(self.config.model_output):
                        os.makedirs(self.config.model_output)
                    saver.save(sess, self.config.model_output)
                    best_score = f1
                    self.logger.info("- new best score!")

                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                        self.logger.info("- early stopping {} epochs without improvement".format(
                            nepoch_no_imprv))
                        break

    def test(self, test, target_output):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            self.logger.info("Testing model over the test set")
            saver.restore(sess, self.config.model_output)
            acc, f1 = self.evaluate(sess, test, target_output)
            self.logger.info("- test acc {:04.2f} - f1 {:04.2f}".format(100 * acc, 100 * f1))

    def interactive_shell(self, tags, processing_word):
        idx_to_tag = {idx: tag for tag, idx in tags.items()}
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.config.model_output)
            self.logger.info("Interactive Shell Started\nType 'quit' to quit the shell")
            while True:
                try:
                    sentence = input("input> ")
                    words_raw = sentence.strip().split(" ")

                    if words_raw == ["quit"]:
                        break

                    words = [processing_word(w) for w in words_raw]
                    if type(words[0]) == tuple:
                        words = zip(*words)
                    pred_ids, _ = self.predict_batch(sess, [words])
                    preds = [idx_to_tag[idx] for idx in list(pred_ids[0])]
                    print_sentence(logger=self.logger, data={"x": words_raw, "y": preds})

                except Exception:
                    pass
