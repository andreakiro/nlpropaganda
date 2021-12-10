from typing import Dict, Iterable, List, MutableMapping, Tuple
import logging
import string
import os
from allennlp.data.tokenizers.token_class import Token

from overrides import overrides

from allennlp.data.dataset_readers.dataset_utils.span_utils import enumerate_spans
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import Field, ListField, TextField, SpanField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.fields.metadata_field import MetadataField

from src.utils import filter_function

logger = logging.getLogger(__name__)

@DatasetReader.register('si-reader')
class SpanIdentificationReader(DatasetReader):
    def __init__(
            self,
            max_span_width: int,
            tokenizer: Tokenizer = SpacyTokenizer(),
            token_indexers: Dict[str, TokenIndexer] = {"tokens": SingleIdTokenIndexer()},
            **kwargs
        ):
            super().__init__(**kwargs)
            self._max_span_width = max_span_width
            self.tokenizer = tokenizer
            self.token_indexers = token_indexers
    
    @overrides
    def _read(self, file_path) -> Iterable[Instance]:
        articles_dir = file_path

        with os.scandir(articles_dir) as articles:
            for article in articles:
                if not article.name.endswith(".txt"):
                    continue
                content: str = None
                gold_spans: List[Tuple[int, int]] = None

                # logger.info(f"Reading {article.name[:-3]}")

                with open(article, 'r') as file:
                    content = file.read()

                if "test" not in articles_dir:
                    gold_spans = []
                    with open(os.path.join(articles_dir, "labels", article.name[:-3] + 'task-si.labels'), 'r') as lines:
                        for line in lines:
                            _, span_start, span_end = line.strip().split("\t")
                            gold_spans.append((int(span_start), int(span_end)))
                
                # add assertions
                yield self.text_to_instance(content, gold_spans)
        
        logger.info(f"Finished reading {articles_dir}")
    
    @overrides
    def text_to_instance(self, content: str, gold_spans: List[Tuple[int, int]] = None) -> Instance:
        fields: Dict[str, Field] = {}
        metadata: Dict[str, int] = {}

        # Create the TextField for the article
        tokens: List[Token] = self.tokenizer.tokenize(content)
        article_field = TextField(tokens, self.token_indexers)
        fields["content"] = article_field

        # Add gold spans to instance
        if gold_spans is not None:
            gold_spans_in_token_space = self._map_to_token_space(content, gold_spans, tokens)
            list_gold_span_fields: List[SpanField] = [SpanField(start, end, article_field) for start, end in gold_spans_in_token_space]
            dummy : SpanField = SpanField(-1, -1, article_field).empty_field()
            fields["gold_spans"] = ListField(list_gold_span_fields) if len(list_gold_span_fields) != 0 else ListField([dummy]).empty_field()
            metadata["num_gold_spans"] = len(list_gold_span_fields)

        # Extract article spans and add them to instance
        spans = enumerate_spans(tokens, max_span_width=self._max_span_width, filter_function=filter_function)
        list_span_fields: List[SpanField] = [SpanField(start, end, article_field) for start, end in spans]
        metadata["num_spans"] = len(list_span_fields)
        
        #TODO Prune spans, by using heuristics
        fields["all_spans"] = ListField(list_span_fields)
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

        gold_spans_token_space = [(mapping[start], mapping[end]) for start, end in gold_spans]

        # gold_spans_tokens = [self.tokenizer.tokenize(content[start: end + 1]) for start, end in gold_spans]
        # gold_spans_tokens_in_token_space = [tokens[start: end + 1] for start, end in gold_spans_token_space]

        # logger.info(f"gold spans tokens\n\n{gold_spans_tokens}\n")
        # logger.info(f"gold spans tokens computer\n\n{gold_spans_tokens_in_token_space}\n")

        return gold_spans_token_space