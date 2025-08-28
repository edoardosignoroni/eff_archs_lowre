# -*- coding: utf-8 -*-

import re

from fairseq.data.encoders import register_tokenizer
from fairseq.dataclass import FairseqDataclass



@register_tokenizer("spmdetok", dataclass=FairseqDataclass)
class spmdetokTokenizer(object):
    """
    Detokenize the same way as the sed sript:
    """
    def __init__(self, *unused):
        self.space_tok = re.compile(r"\s+")
        self.sep = re.compile(r'▁')

  
    def encode(self, x: str) -> str:
        return self.space_tok.sub(" ", x)

    def decode(self, x: str) -> str:
        x = x.replace(' ', '')
        x = x.replace('▁', ' ')
        return x
