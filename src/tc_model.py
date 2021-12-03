import logging
from typing import Dict, Optional

import torch
from sklearn.metrics import r2_score
from allennlp.models.model import Model

logger = logging.getLogger(__name__)

@Model.register("technique-classifier")
class TechniqueClassifier(Model):
    """
    This model implements the propaganda technique classification
    from a given set of text spans identified as propagandist.

    # Paramters

    input_size: `int` required.
        Input size to our classification model i.e. spans embedding size.
    output_size: `int` required.
        Output size to our classification i.e. number of propaganda techniques.
    classifier: `torch.nn` optional.
        PyTorch Neural Network model to perform classification task.
    """
    def __init__(self, 
        input_size: int,
        output_size: int,
        classifier: torch.nn = None,
        serialization_dir: Optional[str] = None, # May be clean to use
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        if classifier is None:
            self._classifier = torch.nn.Linear(input_size, output_size)
        else:
            self._classifier = classifier

    def forward(
        self, # type: ignore
        si_spans: torch.IntTensor,
        gold_spans: torch.IntTensor = None,
        gold_labels: torch.IntTensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        # Parameters
        si_spans: `torch.IntTensor` required.
            Batch of spans on which to perform training or prediction.
        gold_spans: `torch.IntTensor` optional.
            Batch of gold spans used to perform model training. 
        gold_labels: `torch.IntTensor` optional.
            Batch of gold_spans labels used to perform model training.

        # Returns 
        An output dictionary consisting of:
        technique: `torch.IntTensor` always
            The model prediction on the batch of given SpanFields.
        loss: `torch.FloatTensor` optional
            A scalar loss to be optimised.
        """
        if gold_spans is not None:
            assert gold_labels is not None, "Must provide gold spans and corresponding labels."

        logits = self._classifier(si_spans)
        technique_probs = torch.softmax(logits)

        # Not sure of that yet
        max_idx = torch.argmax(technique_probs, dim=1)
        technique_preds = torch.FloatTensor(technique_probs.shape).zero_()
        technique_preds.scatter_(0, max_idx, 1)

        
        output_dict = {
            "technique_preds": technique_preds,
            "technique_probs": technique_probs,
        }

        if gold_spans is not None:
            labels = self._get_labels(si_spans, gold_spans, gold_labels)
            output_dict["loss"] = self._compute_loss(labels, technique_preds)
        
        return output_dict

    def get_metrics(self) -> Dict[str, float]:
        return None # TODO
    
    def _compute_loss(
        self, 
        labels: torch.IntTensor,
        predictions: torch.IntTensor,
    ) -> float:
        """
        # Parameters
        label: `torch.IntTensor` required.
            Ground truth label (propaganda technique) for a given span.
        prediction: `torch.IntTensor` required.
            Model propaganda technique prediction for a given span.
        
        # Returns
        score: `float` always
            sklean R2 score `float`
        """
        return r2_score(labels, predictions)
    
    def _get_labels(
        self,
        si_spans: torch.IntTensor,
        gold_spans: torch.IntTensor,
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
        return None