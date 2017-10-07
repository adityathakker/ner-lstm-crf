import os
from general_utils import get_logger


class Config(object):
    def __init__(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.logger = get_logger(self.log_path)

    output_path = "results/crf/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"

    dim = 500
    dim_char = 100
    glove_filename = "data/glove.txt"
    trimmed_filename = "data/glove.trimmed.npz"

    dev_filename = "data/hindi.dev.conll_format"
    test_filename = "data/hindi.test.conll_format"
    train_filename = "data/hindi.train.conll_format"
    max_iter = None  # if not None, max number of examples

    words_filename = "data/words.txt"
    tags_filename = "data/tags.txt"
    chars_filename = "data/chars.txt"

    train_embeddings = True
    nepochs = 100
    dropout = 0.5
    batch_size = 128
    lr = 0.001
    lr_decay = 0.99
    nepoch_no_imprv = 10

    hidden_size = 700
    char_hidden_size = 200
