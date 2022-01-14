from typing import Dict, Iterable, List, Tuple
import logging
import os
import string

from overrides import overrides
from src.utils import filter_function, label_to_int_alt, get_not_propaganda

import torch
import numpy as np

from allennlp.data.tokenizers.token_class import Token
from allennlp.data.dataset_readers.dataset_utils.span_utils import enumerate_spans
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import Field, ListField, TextField, SpanField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.fields.label_field import LabelField
from allennlp.data.fields.metadata_field import MetadataField

logger = logging.getLogger(__name__)

@DatasetReader.register('tc-reader-alt')
class TechniqueClassificationReaderAlt(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = SpacyTokenizer(),
        token_indexers: Dict[str, TokenIndexer] = {"tokens": SingleIdTokenIndexer()},
        **kwargs
    ):
        super().__init__(**kwargs)
        self._tokenizer = tokenizer
        self._token_indexer = token_indexers

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        articles_dir = file_path
        with os.scandir(articles_dir) as articles:
            for article in articles:
                if not article.name.endswith('.txt'):
                    continue
                content: str = None
                spans: List[Tuple[int, int]] = [] 
                labels: List[int] = None

                with open(article, 'r') as file:
                    content = file.read()

                if "test" in articles_dir:
                    raise Exception('Undefined', 'We dont have the data!')

                if "test" not in articles_dir:
                    labels = []
                    with open(os.path.join(articles_dir, 'labels', article.name[:-3] + 'task-flc-tc.labels'), 'r') as lines:
                        for line in lines:
                            _, label, span_start, span_end = line.strip().split("\t")
                            spans.append((int(span_start), int(span_end)))
                            labels.append(label_to_int_alt(label))
                    if "dev" in articles_dir:
                        labels = None
                
                yield self.text_to_instance(content, article.name[7:-4], spans, labels)
    
    @overrides
    def text_to_instance(
        self, content: str, 
        article_id: int,
        spans: List[Tuple[int, int]] = None,
        labels: List[str] = None,
    ) -> Instance:
        fields: Dict[str, Field] = {}
        metadata: Dict[str, int] = {}

         # Create the TextField for the article
        tokens = self._tokenizer.tokenize(content)
        article_field = TextField(tokens, self._token_indexer)
        fields["content"] = article_field

        # Add gold labels to instance
        if labels is not None:
            labs: List[LabelField] = [LabelField(x, skip_indexing=True) for x in labels]
            dummy: LabelField = LabelField(-1, skip_indexing=True).empty_field()
            fields["gold_labels"] = ListField(labs) if len(labs) > 0 else ListField([dummy]).empty_field()
        
        # Add spans to instance
        spans_token_space = self._map_to_token_space(content, spans, tokens)
        span_field: List[SpanField] = [SpanField(start, end, article_field) for start, end in spans_token_space]
        dummy: SpanField = SpanField(-1, -1, article_field).empty_field()
        fields["spans"] = ListField(span_field) if len(span_field) > 0 else ListField([dummy]).empty_field()

        metadata["article_id"] = article_id
        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    def _map_to_token_space(
        self,
        content: str,
        gold_spans: List[Tuple[int, int]],
        tokens: List[Token]
    ):
        mapping = {}

        idx = 0
        token_id = 0
        cur_token_size = len(tokens[token_id].text)
        gold_spans_flattened = [bound for span in gold_spans for bound in span]

        for i, char in enumerate(content):
            if i in gold_spans_flattened:
                if char in string.whitespace:
                    mapping[i] = token_id - 1
                else:
                    mapping[i] = token_id

            if char in string.whitespace:
                if i in gold_spans_flattened:
                    mapping[i] = token_id - 1
            else:
                if i in gold_spans_flattened:
                    mapping[i] = token_id

            if char not in string.whitespace:
                if idx < cur_token_size - 1:
                    idx += 1
                else:
                    idx = 0
                    token_id += 1
                    if token_id < len(tokens):
                        cur_token_size = len(tokens[token_id].text)

        return [(mapping[start], mapping[end]) for start, end in gold_spans]
