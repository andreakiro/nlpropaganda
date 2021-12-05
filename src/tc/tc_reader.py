from typing import Dict, Iterable, List, Tuple
import logging
import string
import os

from overrides import overrides

import torch

from allennlp.data.dataset_readers.dataset_utils.span_utils import enumerate_spans
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import Field, ListField, TextField, SpanField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer

logger = logging.getLogger(__name__)

@DatasetReader.register('tc-reader')
class TechniqueClassificationReader(DatasetReader):
    """
    Read our input data for technique classification task.
    Either for training or for testing purposes. Returns a 
    `Dataset` where the `Instances` have at least two fields
    and a third one if we are dealing with a training session.
    
    fields["articles-content"]: `TextField` always.
        The entire article content in a TextField.
    fields["si-spans"]: `ListField[SpanField] always.
        All spans retrieved as potential propaganda from si-model.
    fields["gold-spans"]: `SequenceLabelField[List[int], ListField[SpanField]] when training.
        All spans from the training data with their corresponding technique label annotation.

    # Parameters
    task: `str` required. TODO: Not anymore.
        To decide whether to prepare a training or testing Dataset.
        Must be either "training" or "testing" otherwise an error is raised.
    si_model: `torch.nn` required.
        Model from which we'll recover the potential propaganda spans.
    max_span_width : `int` required.
        Maximum span width heuristic used for pruning. TODO: Deal with that.
    data_directory_path : `str` required.
        Path where the data is located. Either test or training one.
    labels_file_path: `str` optional.
        Path where the labels are located. Given only when training.
    tokenizer: `Tokenizer` optional.
        Text tokenizer used on the data. Default is the SpacyTokenizer.
    token_indexer: `Dict[str, TokenIndexer]` optional.
        Token indexer used on the data. Default is SingleIdTokenIndexer.
    """
    def __init__(
        self,
        si_model: torch.nn,
        max_span_width: int,
        articles_dir_path: str,
        labels_file_path: str = None,
        tokenizer: Tokenizer = SpacyTokenizer(),
        token_indexer: Dict[str, TokenIndexer] = {"tokens": SingleIdTokenIndexer()},
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if labels_file_path is not None:
            assert labels_file_path.endswith(".labels"), ".labels file expected for training."
            self._task = "training"
        else:
            self._task = "testing"
        
        self._si_model = si_model
        self._max_span_width = max_span_width
        self.articles_dir_path = articles_dir_path
        self._labels_file_path = labels_file_path
        self._tokenizer = tokenizer
        self._token_indexer = token_indexer

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # Note: file_path is folder_to_articles_path 
        logger.info(f"Preparing Dataset for {self._task}")
        logger.info(f"Searching for data in {self.articles_dir_path}")

        instances: list[Dict[str, Field]] = []

        logger.info(f"Loading articles content from {self._articles_dir_path}")
        with os.scandir(self._articles_dir_path) as articles:
            for article in articles:
                assert article.name.endswith(".txt"), "Articles must be under txt format."
                with open(article, 'r') as file:
                    content = file.read()
                    instances.append({"article-content": content})

        # add every si-spans predicted from si-model ?

        if self._task == "training":
            logger.info(f"Loading labels from {self._labels_file_path}")
            with open(self._labels_file_path) as lines:
                {}

        # fields: Dict[str, Field] = {}
        # fields["articles-content"] = _get_articles_instances(self)
        # fields["si-spans"] = _get_si_spans_instances(self)

        # if self._task == "training":
        #     fields["gold-spans"] = _get_gold_spans_instances(self)

        # return Iterable[Instance[fields]]
        return None

    @overrides
    def text_to_instance(self, *inputs) -> Instance:
        return super().text_to_instance(*inputs)

    def _get_article_instances(self) -> Field:
        return Field

    def _get_si_spans_instances(self) -> Field:
        return Field

    def _get_gold_spans_instances(self) -> Field:
        return None

    
# 1) Lire tous les articles
# 2) Lire tous les labels files
# 3) Créer une méthode filter()
#   filter(si-model, enumerated_spans) -> filtered_spans
# 4) Créer les instances i.e. :
#   Si training: 
#       TextField (article)
#       ListField[SpanField]
#       ListField[LabeledSequenceField] (spans labelisés)
#           LabelSequenceField := Tuple[SpanField, int]
#   Si test:
#       TextField (article)
#       ListField[SpanField]
