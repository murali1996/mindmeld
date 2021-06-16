# # -*- coding: utf-8 -*-
# #
# # Copyright (c) 2015 Cisco Systems, Inc. and others.  All rights reserved.
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #     http://www.apache.org/licenses/LICENSE-2.0
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
#
# import json
# import logging
# import os
# from abc import abstractmethod
# from typing import Dict
#
# import torch.nn as nn
#
# from .encoders import TextsEncoder
#
# logger = logging.getLogger(__name__)
#
#
# class FeaturizerConfig:
#     """A value object representing a featurizer configuration
#
#         Attributes:
#             model_type (str): The name of the model type. Will be used to find the
#                 model class to instantiate.
#             model_settings (dict): Settings specific to the model type specified.
#             params (dict): Params to determine various settings in the underlying featurizer init()
#     """
#
#     __slots__ = [
#         "model_type",
#         "model_settings",  # for model layers' creation time;includes n_filters, filter_lengths, etc
#         "params",  # for model fitting time;includes n_epochs, optimizer_type, loss_type, etc.
#     ]
#
#     def __init__(self, model_type=None, model_settings=None, params=None):
#         for arg, val in {"model_type": model_type, }.items():
#             if val is None:
#                 raise TypeError("__init__() missing required argument {!r}".format(arg))
#
#         self.model_type = model_type
#         self.model_settings = model_settings
#         self.params = params
#
#     def __repr__(self):
#         args_str = ", ".join("{}={!r}".format(key, getattr(self, key)) for key in self.__slots__)
#         return "{}({})".format(self.__class__.__name__, args_str)
#
#     def get_model_settings(self, attr, default):
#         if attr not in self.model_settings:
#             self.model_settings.update({attr: default})
#         return self.model_settings[attr]
#
#     def to_dict(self):
#         result = {}
#         for attr in self.__slots__:
#             result[attr] = getattr(self, attr)
#         return result
#
#     @classmethod
#     def from_model_config(cls, model_config):
#         config = model_config.to_dict()
#         config.pop("example_type", None)
#         config.pop("label_type", None)
#         config.pop("train_label_set", None)
#         config.pop("test_label_set", None)
#         config.pop("features", None)
#         config.pop("param_selection", None)
#         return cls(**config)
#
#     @classmethod
#     def load_config(self, load_folder) -> FeaturizerConfig:
#         with open(os.path.join(load_folder, "featurizer_config.json"), "r") as fp:
#             config_dict = json.load(fp)
#             fp.close()
#         return cls(**config_dict)
#
#     def dump_config(self, dump_folder) -> None:
#         os.makedirs(os.path.dirname(dump_folder), exist_ok=True)
#         with open(os.path.join(dump_folder, "featurizer_config.json"), "w") as fp:
#             json.dump(self.config.to_dict(), fp, indent=4)
#             fp.close()
#
#
# class Featurizer(nn.Module):
#
#     def __init__(self, config: FeaturizerConfig, **kwargs):
#         super().__init__()
#         self.config = config
#         self.resource_loader = kwargs.get("resource_loader")
#
#     @property
#     def device(self):
#         return next(self.parameters()).device
#
#     @abstractmethod
#     def initialize_resources(self, **kwargs) -> None:
#         """
#         This method initializes resources of the featurirer class
#             and resources of any associated tokenizer_encoder(s)
#         """
#         msg = f"Deriving class {self.__class__.__name__} need to implement this"
#         raise NotImplementedError(msg)
#
#     @abstractmethod
#     def dump_resources(self, dump_folder) -> None:
#         """
#         This method dumps the featurizer config, associated tokenizer_encoder resources,
#             and model state dict
#         """
#         msg = f"Deriving class {self.__class__.__name__} need to implement this"
#         raise NotImplementedError(msg)
#
#     @abstractmethod
#     def load_resources(self, load_folder) -> None:
#         """
#         This method loads associated tokenizer_encoder resources, and model state dict
#         """
#         msg = f"Deriving class {self.__class__.__name__} need to implement this"
#         raise NotImplementedError(msg)
#
#     # @abstractmethod
#     # def train(self, examples, labels) -> None:
#
#
# class CharCnnTextFeaturizer(Featurizer):
#     """
#     CNN model that runs convolutions on character level embeddings sequence of each input.
#     This leads to one representation per input and hence the output features consist
#     of `text_features` and not `tagger_features`.
#     """
#
#     def initialize_resources(self, resource_loader, examples=None, labels=None, **kwargs) -> None:
#
#         if not examples and not labels:
#             # due to the way of implementation of Classifier class and its load method,
#             #   this method is called after loading the model, in which case, everything is already
#             #   initialized as for Featurizers, load() also includes initialize_resources()
#             return
#
#         self.encoder = TextsEncoder().initialize_resources(
#             texts=examples, tokenizer_type="char-tokenizer")
#         n_tokens = self.encoder.get_vocab_size
#
#         self.config.get_params()
#
#         n_tokens = self.configs.get("n_tokens")
#         emb_dim = self.configs.get("emb_dim")
#         padding_idx = self.configs.get("padding_idx")
#         filter_lens = self.configs.get("filter_lens")
#         n_filters = self.configs.get("n_filters")
#
#         self.embeddings = nn.Embedding(n_tokens, emb_dim, padding_idx=padding_idx)
#         self.embeddings.weight.requires_grad = True
#
#         # Unsqueeze [BS, MAXSEQ, EMDDIM] as [BS, 1, MAXSEQ, EMDDIM] and send as input
#         self.convmodule = nn.ModuleList()
#         for length, n in zip(filter_lens, n_filters):
#             self.convmodule.append(
#                 nn.Sequential(
#                     nn.Conv2d(1, n, (length, emb_dim), padding=(length - 1, 0),
#                               dilation=1, bias=True, padding_mode='zeros'),
#                     nn.ReLU(),
#                 )
#             )
#         # each conv outputs [BS, n_filters, MAXSEQ, 1]
#
#         self.dropout = nn.Dropout(p=0.3)
#         self.outdim = sum(n_filters)
#
#         print("CNNMeanTokens model initialized")
#
#     def dump_resources(self, dump_folder) -> None:
#         raise NotImplementedError
#
#     def load_resources(self, load_folder) -> None:
#         raise NotImplementedError
#
#     raise NotImplementedError
#
#
# class WordCnnTextFeaturizer(Featurizer):
#     def __init__(self, config):
#         super().__init__(config)
#
#     raise NotImplementedError
#
#
# class WordLstmTextFeaturizer(Featurizer):
#     def __init__(self, config):
#         super().__init__(config)
#
#     raise NotImplementedError
#
#
# class CharCnnTaggerFeaturizer(Featurizer):
#     """
#     CNN model that runs convolutions on character level embeddings for each word of each input.
#     This leads to one representation per input and hence the output features consist
#     of `tagger_features` and not `text_features`.
#     """
#
#     def __init__(self, config):
#         super().__init__(config)
#
#     raise NotImplementedError
#
#
# class WordLstmTaggerFeaturizer(Featurizer):
#     def __init__(self, config):
#         super().__init__(config)
#
#     raise NotImplementedError
#
#
# class AutoFeaturizer:
#     CLASSIFIER_TYPES = {
#         "text": {
#             "char-cnn": CharCnnTextFeaturizer,
#             "word-cnn": WordCnnTextFeaturizer,
#             "word-lstm": WordLstmTextFeaturizer,
#         },
#         "tagger": {
#             "char-cnn": CharCnnTaggerFeaturizer,
#             "word-lstm": WordLstmTaggerFeaturizer
#         }
#     }
#
#     @classmethod
#     def from_config(cls, config: Dict):
#         featurizer_config = FeaturizerConfig.from_model_config(config)
#         model_type = featurizer_config.model_type
#         classifier_type = featurizer_config.model_settings.get("classifier_type")
#
#         if not (model_type and model_type in AutoFeaturizer.CLASSIFIER_TYPES):
#             msg = f"Invalid 'model_type': {model_type}. Expected one of {[*model_type.keys()]}"
#             raise ValueError(msg)
#
#         if not (classifier_type and classifier_type in AutoFeaturizer.CLASSIFIER_TYPES[model_type]):
#             msg = f"Invalid 'classifier_type': {classifier_type}. " \
#                   f"Expected one of {[*AutoFeaturizer.CLASSIFIER_TYPES.keys()]}"
#             raise ValueError(msg)
#
#         return AutoFeaturizer.CLASSIFIER_TYPES[model_type][classifier_type](featurizer_config)
#
#     @classmethod
#     def from_path(cls, load_folder):
#         featurizer_config = FeaturizerConfig.load_config(load_folder)
#         model_type = featurizer_config.model_type
#         classifier_type = featurizer_config.model_settings.get("classifier_type")
#         return AutoFeaturizer.CLASSIFIER_TYPES[model_type][classifier_type](featurizer_config)
