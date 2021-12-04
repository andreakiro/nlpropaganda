from typing import Dict, Iterable, List, Tuple
import logging
import string
import os

from overrides import overrides

from allennlp.data.dataset_readers.dataset_utils.span_utils import enumerate_spans
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import Field, ListField, TextField, SpanField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer

logger = logging.getLogger(__name__)

@DatasetReader.register('tc-reader')
class SpanIdentificationReader(DatasetReader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)