#!/usr/bin/python
# -*- coding: utf-8 -*-

from src.si.si_reader import SpanIdentificationReader
from src.si.si_metric import SpanIdenficationMetric
from src.si.si_model import SpanIdentifier
from src.si.si_predictor import SpanIdentificationPredictor

from src.tc.tc_reader import TechniqueClassificationReader
from src.tc.tc_model import TechniqueClassifier
from src.tc.tc_predictor import TechniqueClassificationPredictor

from src.tc.alternative.tc_reader_alt import TechniqueClassificationReaderAlt
from src.tc.alternative.tc_model_alt import TechniqueClassifierAlt
from src.tc.alternative.tc_predictor_alt import TechniqueClassificationPredictorAlt
from src.tc.alternative.tc_predictor_alt_demo import TechniqueClassificationPredictorAltDemo