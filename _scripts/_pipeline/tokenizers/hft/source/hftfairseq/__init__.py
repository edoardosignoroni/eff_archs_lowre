# -*- coding: utf-8 -*-

import re

from fairseq.data.encoders import register_tokenizer
from fairseq.dataclass import FairseqDataclass



@register_tokenizer("hftdetok", dataclass=FairseqDataclass)
class HFTdetokTokenizer(object):
    """
    Detokenize the same way as the sed sript:
    """
    def __init__(self, *unused):
        self.space_tok = re.compile(r"\s+")
        self.caps = re.compile(r'𐊣([^𐊼]*)𐊼')
        self.shift = re.compile(r'𐋇(.)')
        self.upper = lambda m: m.group(1).upper()


    def encode(self, x: str) -> str:
        return self.space_tok.sub(" ", x)

    def decode(self, x: str) -> str:
        x = x.replace(' ', '')
        x = x.replace('𐋇¦', '¦𐋇').replace('𐊣¦', '¦𐊣').replace('¦𐊼','𐊼¦')
        x = x.replace('¦¦', ' ').replace('¦', '').replace('▁', ' ')
        x = self.caps.sub(self.upper, x)
        x = self.shift.sub(self.upper, x)
        return x
