from typing import Dict, Iterable, List, Tuple
import logging
import string
import os

from overrides import overrides

import numpy as np

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
            data_directory_path: str,
            max_span_width: int,
            tokenizer: Tokenizer = SpacyTokenizer(),
            token_indexers: Dict[str, TokenIndexer] = {"tokens": SingleIdTokenIndexer()},
            **kwargs
        ):
            super().__init__(**kwargs)
            self._data_directory_path = data_directory_path
            self._max_span_width = max_span_width
            self.tokenizer = tokenizer
            self.token_indexers = token_indexers

    @overrides
    def text_to_instance(self, text: str, gold_spans: List[Tuple[int, int]] = None) -> Instance:
        fields: Dict[str, Field] = {}

        # Create the TextField for the article
        tokens = self.tokenizer.tokenize(text)
        article_field = TextField(tokens, self.token_indexers)
        fields["text"] = article_field

        # Mapping gold spans to the token space
        gold_spans_starts = [start for start, _ in gold_spans]
        gold_spans_tokens = [self.tokenizer.tokenize(text[start:end]) for start, end in gold_spans]
        gold_spans_in_token_space = []
        text_index = 0
        for token_index, token in enumerate(tokens):
            if text_index in gold_spans_starts:
                logger.info(f"START OF PROPAGANDAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                # assert gold_spans_tokens[len(gold_spans_in_token_space)] in tokens[token_index:token_index+len(gold_spans_tokens[len(gold_spans_in_token_space)])], "Mismatch between character and token space"
                gold_spans_in_token_space.append(token_index)
            
            logger.info(f"text index before: {text_index}")
            logger.info(f"token index: {token_index}")
            logger.info(f"token: {token.text}")
            text_index += len(token.text)
            logger.info(f"text index after: {text_index}")
            logger.info(f"text after this: {text[text_index: text_index+10]}")

            while text_index < len(text) and text[text_index] in string.whitespace:
                text_index += 1
            logger.info(f"text index after whitespace: {text_index}")
        
        gold_spans_in_token_space = [(start, start + len(tokens) - 1) for start, tokens in zip(gold_spans_in_token_space, gold_spans_tokens)]

        # gold_spans_in_token_space = [[x for x in range(len(tokens)) if [token.text for token in tokens[x:x+len(span_tokens)]] == [token.text for token in span_tokens]] for span_tokens in gold_spans_tokens]
        # Debug
        logger.info(f"text\n{text}")
        logger.info(f"tokens\n{tokens}")
        logger.info(f"gold spans text\n{[text[start:end] for start, end in gold_spans]}")
        logger.info(f"gold spans tokens\n{gold_spans_tokens}")
        logger.info(f"gold spans in token space\n{gold_spans_in_token_space}")
        logger.info(f"gold spans in token space in tokens\n{[tokens[start:end+1] for start, end in gold_spans_in_token_space]}")
        # assert np.all([len(starts) == 1 for starts in gold_spans_in_token_space]), "Failed conversion of span in token space"
        # SpanField's bounds are inclusive
        

        # Add gold spans to instance
        list_gold_span_fields: List[SpanField] = [SpanField(start, 
                                                            end, 
                                                            article_field) 
                                                            for start, end in gold_spans_in_token_space]
        fields["gold_spans"] = ListField(list_gold_span_fields)

        # Extract article spans and add them to instance
        spans = enumerate_spans(tokens, max_span_width=self._max_span_width)
        list_span_fields: List[SpanField] = [SpanField(start, 
                                                       end, # enumerate_spans returns spans with inclusive boundaries
                                                       article_field) 
                                                       for start, end in spans]
        #TODO: prune spans, by using heuristics and/or a model
        fields["spans"] = ListField(list_span_fields)

        return Instance(fields)

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:

        logger.info(f"Searching for data in {self._data_directory_path}")

        assert file_path.endswith(".labels"), ".labels file expected"

        with open(os.path.join(self._data_directory_path, file_path), 'r') as lines:
            logger.info(f"Loading labels from {os.path.join(self._data_directory_path, file_path)}")

            current_article_id: str = None
            spans: List[Tuple[int, int]] = []
            for line in lines:
                article_id, span_start, span_end = line.strip().split("\t")

                span_start = int(span_start)
                span_end = int(span_end)

                # Deal with first article
                if current_article_id == None:
                    current_article_id = article_id

                if article_id != current_article_id:
                    # New article, flush old article data
                    yield self._instanciate_article_data(current_article_id, spans)
                    spans.clear()
                    current_article_id = article_id

                # Remember spans for current article
                spans.append((span_start, span_end))

            # Deal with last article
            yield self._instanciate_article_data(current_article_id, spans)

    def _instanciate_article_data(self, article_id, spans):
        text = None
        with os.scandir(self._data_directory_path) as entries:
            for entry in entries:
                if article_id in entry.name and entry.name.endswith(".txt"):
                    with open(entry, 'r') as file:
                        text = file.read()
        assert text is not None, f"Article {article_id} not found"

        return self.text_to_instance(text, spans)