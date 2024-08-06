# coding=utf-8

"""Tokenization classes."""

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import logging
import os
from io import open
from layers.snippets import truncate_sequences

logger = logging.getLogger(__name__)


def load_vocab(vocab_file):
    """
    load vocab dict file to dict object"""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


class ICD_Tokenizer(object):
    def __init__(
            self,
            vocab_file,  # The path to the vocabulary file, vocabulary file is dict
            unk_token="[UNK]", # Special tokens used for unknown tokens
            sep_token="[SEP]", # Given one sentence, use [SEP] in the end; given two sentences, use [SEP] in the end of two sentences
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]"
    ):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'.".format(vocab_file))
        # load the vocabulary from vocab_file file and create a mapping from token IDs to tokens.
        self.vocab = load_vocab(vocab_file)
        # print(self.vocab)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token

    def tokenize(self, text):
        if self.cls_token is not None:
            text.insert(0, self.cls_token)
        if self.sep_token is not None:
            text.append(self.sep_token)
        return text

    def convert_tokens_to_ids(self, tokens):
        """ Convert tokens to their corresponding token IDs """
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        return ids

    def convert_ids_to_tokens(self, ids):
        """ Convert token IDs to their corresponding tokens """
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    def encode(
            self,
            seq_icds,
            max_position=512,
            truncate_from='right'
    ):
        """
        Encode text (ICD codes) into token IDs
        """
        seq_icds = self.tokenize(seq_icds)
        if len(seq_icds) > max_position:
            if truncate_from == 'right':
                truncate_index = -2
            elif truncate_from == 'left':
                truncate_index = 1
            else:
                truncate_index = truncate_from
            truncate_sequences(max_position, truncate_index, seq_icds) # inplace truncation
        seq_tokens = self.convert_tokens_to_ids(seq_icds)
        return seq_tokens

