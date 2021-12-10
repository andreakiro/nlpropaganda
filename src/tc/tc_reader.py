from typing import Dict, Iterable, List, Tuple
import logging
import os
from allennlp.models.model import Model

from overrides import overrides

from src.utils import filter_function, label_to_int, get_not_propaganda

import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder

from allennlp.data.dataset_readers.dataset_utils.span_utils import enumerate_spans
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import Field, ListField, TextField, SpanField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.models.archival import Archive
from allennlp.data.fields.label_field import LabelField

from src.si.si_model import SpanIdentifier


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
        max_span_width: int,
        si_model: Model,
        tokenizer: Tokenizer = SpacyTokenizer(),
        token_indexers: Dict[str, TokenIndexer] = {"tokens": SingleIdTokenIndexer()},
        **kwargs
    ):
        super().__init__(**kwargs)
        self._si_model = si_model
        self._max_span_width = max_span_width
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
                gold_label_spans: List[Tuple[str, int, int]] = None

                with open(article, 'r') as file:
                    content = file.read()

                if "test" not in articles_dir:
                    gold_label_spans = []
                    with open(os.path.join(articles_dir, 'labels', article.name[:-3] + 'task-flc-tc.labels'), 'r') as lines:
                        for line in lines:
                            _, label, span_start, span_end = line.strip().split("\t")
                            gold_label_spans.append((label, int(span_start), int(span_end)))
                
                # add assertions
                yield self.text_to_instance(content, gold_label_spans)
        
        logger.info(f'Finished reading {articles_dir}')
    
    @overrides
    def text_to_instance(
        self, content: str, 
        gold_label_spans: List[Tuple[str, int, int]] = None
    ) -> Instance:
        fields: Dict[str, Field] = {}

         # Create the TextField for the article
        tokens = self._tokenizer.tokenize(content)
        article_field = TextField(tokens, self._token_indexer)
        fields["content"] = article_field

        # Extract article spans and add them to instance
        all_spans = enumerate_spans(tokens, max_span_width = self._max_span_width, filter_function=filter_function)
        spans_field = ListField([SpanField(start, end, article_field) for start, end in all_spans])

        # Call SI model to get si-spans
        inst = Instance({'content': article_field, 'all_spans': spans_field})
        si_output = self._si_model.forward_on_instance(inst)
        si_spans = torch.as_tensor(si_output["all-spans"])
        si_probs = torch.as_tensor(si_output["probs-spans"])

        mask = si_probs >= 0.65
        mask = torch.stack((mask, mask), dim=1).reshape(mask.shape[0], 2)
        filtered_spans = torch.masked_select(si_spans, mask).reshape(-1, 2)


        if filtered_spans.shape[0] == 0:
            argmax = torch.argmax(si_probs)
            filtered_spans = si_spans[argmax].reshape(-1, 2)

        logger.info(f"FILTERED: {filtered_spans.shape}")

        # Add si-spans to our field dict
        fields["si_spans"] = ListField([SpanField(start.item(), end.item(), article_field) for start, end in filtered_spans])

        if gold_label_spans is not None:
            # Add gold-spans to our field dict
            labels = self._get_labels(filtered_spans, gold_label_spans)
            fields["gold_labels"] = ListField([LabelField(label, skip_indexing=True) for label in labels])

        return Instance(fields)

    def _get_labels(
        self,
        si_spans: torch.IntTensor,
        gold_label_spans: torch.IntTensor,
    ) -> torch.IntTensor:
        """
        # Parameters
        si_spans: `torch.IntTensor` required.
            Batch of SpanFields (?) on which to perform training or prediction.
        gold_spans: `torch.IntTensor` optional.
            Batch of SpanFields corresponding labels when training the model.
        
        # Returns
        labels: `torch.IntTensor` always
            Labels of span s for every span s in si_spans appearing in gold_spans
            Label "no propaganda" for every span s in si_spans not appearing in gold_spans
        """
        labels: List[int] = []
        
        for span in si_spans:
            labelled = False
            for tup in gold_label_spans:
                if span[0] == tup[1] and span[1] == tup[2]: # smarter ? achtung mapping int->str
                    labels.append(label_to_int(tup[0]))
                    labelled = True
                    break
            if not labelled:
                labels.append(get_not_propaganda())
        
        #return torch.as_tensor(LabelEncoder().fit_transform(labels))
        return labels