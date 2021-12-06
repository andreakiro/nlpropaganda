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

                logger.info(f"Reading {article.name[:-3]}")

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
    def text_to_instance(self, text: str, gold_spans: List[Tuple[int, int]] = None) -> Instance:
        fields: Dict[str, Field] = {}

        # Create the TextField for the article
        tokens = self.tokenizer.tokenize(text)
        article_field = TextField(tokens, self.token_indexers)
        fields["text"] = article_field

        # Add gold spans to instance
        if gold_spans is not None:
            gold_spans_in_token_space = self._map_to_token_space(text, gold_spans, tokens)
            list_gold_span_fields: List[SpanField] = [SpanField(start, end, article_field) for start, end in gold_spans_in_token_space]
            fields["gold_spans"] = ListField(list_gold_span_fields) if len(list_gold_span_fields) != 0 else ListField.empty_field

        # Extract article spans and add them to instance
        spans = enumerate_spans(tokens, max_span_width=self._max_span_width)
        list_span_fields: List[SpanField] = [SpanField(start, end, article_field) for start, end in spans]
        
        #TODO Prune spans, by using heuristics
        fields["spans"] = ListField(list_span_fields)

        return Instance(fields)

    def _map_to_token_space(self, text: str, gold_spans, tokens):
        # Mapping gold spans to the token space
        gold_spans_starts = [start for start, _ in gold_spans]
        mapping = {}

        text_index = 0
        if text_index in gold_spans_starts:
            mapping[text_index] = 0
        
        for token_index, token in enumerate(tokens):
            
            tmp = text_index
            while text_index < tmp + len(token.text):
                text_index += 1
                if text_index in gold_spans_starts:
                    mapping[text_index] = token_index

            while text_index < len(text) and text[text_index] in string.whitespace:
                text_index += 1
                if text_index in gold_spans_starts:
                    mapping[text_index] = token_index

        gold_spans_tokens = [self.tokenizer.tokenize(text[start:end+1]) for start, end in gold_spans]
        logger.info(f"gold spans tokens\n\n{gold_spans_tokens}\n")

        gold_spans_in_token_space = [mapping[start] for start, _ in gold_spans]
        gold_spans_in_token_space = [(start_token, len(self.tokenizer.tokenize(text[span[0]:span[1]]))) for start_token, span in zip(gold_spans_in_token_space, gold_spans)]

        logger.info(f"gold spans in token space in tokens\n\n{[tokens[start:end+1] for start, end in gold_spans_in_token_space]}\n")
        return gold_spans_in_token_space

    def _map_to_token_space_2(self, text: str, gold_spans, tokens):
        # Mapping gold spans to the token space
        gold_spans_flattened = [bound for span in gold_spans for bound in span]
        mapping = {}

        text_index = 0
        if text_index in gold_spans_flattened:
            mapping[text_index] = 0
        
        for token_index, token in enumerate(tokens):
            
            tmp = text_index
            while text_index < tmp + len(token.text):
                text_index += 1
                if text_index in gold_spans_flattened:
                    mapping[text_index] = token_index

            while text_index < len(text) and text[text_index] in string.whitespace:
                text_index += 1
                if text_index in gold_spans_flattened:
                    mapping[text_index] = token_index

        gold_spans_tokens = [self.tokenizer.tokenize(text[start:end+1]) for start, end in gold_spans]
        logger.info(f"gold spans tokens\n\n{gold_spans_tokens}\n")
        gold_spans_in_token_space = [(mapping[start], mapping[end]) for start, end in gold_spans]
        logger.info(f"gold spans in token space in tokens\n\n{[tokens[start:end+1] for start, end in gold_spans_in_token_space]}\n")
        return gold_spans_in_token_space