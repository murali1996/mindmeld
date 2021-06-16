# -*- coding: utf-8 -*-
#
# Copyright (c) 2015 Cisco Systems, Inc. and others.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains base class for tagger classifiers
"""

import logging

logger = logging.getLogger(__name__)


class Tagger:
    """A class for all sequence tagger models implemented in house.
    It is importent to follow this interface exactly when implementing a new model so that your
    model is configured and trained as expected in the MindMeld pipeline. Note that this follows
    the sklearn estimator interface so that GridSearchCV can be used on our sequence models.
    """

    def __init__(self, **parameters):
        """To be consistent with the sklearn interface, __init__ and set_params should have the same
        effect. We do all parameter setting and validation in set_params which is called from here.

        Args:
            **parameters: Arbitrary keyword arguments. The keys are model parameter names and the
                          values are what they should be set to
        Returns:
            self
        """
        self.set_params(**parameters)

    def __getstate__(self):
        """Returns the information needed pickle an instance of this class. By default, pickling removes
        attributes with names starting with underscores.
        """
        attributes = self.__dict__.copy()
        return attributes

    def fit(self, X, y):
        """Trains the model. X and y are the format of what is returned by extract_features. There is no
        restriction on their type or content. X should be the fully processed data with extracted
        features that are ready to be used to train the model. y should be a list of classes as
        encoded by the label_encoder

        Args:
            X (list): Generally a list of feature vectors, one for each training example
            y (list): A list of classification labels (encoded by the label_encoder, NOT MindMeld
                      entity objects)
        Returns:
            self
        """
        raise NotImplementedError

    def predict(self, X, dynamic_resource=None):
        """Predicts the labels from a feature matrix X. Again X is the format of what is returned by
        extract_features.

        Args:
            X (list): A list of feature vectors, one for each example
        Returns:
            (list of classification labels): a list of predicted labels (in an encoded format)
        """
        raise NotImplementedError

    def get_params(self, deep=True):
        """Gets a dictionary of all of the current model parameters and their values

        Args:
            deep (bool): Not used, needed for sklearn compatibility
        Returns:
            (dict): A dictionary of the model parameter names as keys and their set values
        """
        raise NotImplementedError

    def set_params(self, **parameters):
        """Sets the model parameters. Defaults should be set for all parameters such that a model
        is initialized with reasonable default parameters if none are explicitly passed in.

        Args:
            **parameters: Arbitrary keyword arguments. The keys are model parameter names and the
                          values are what they should be set to
        Returns:
            self
        """
        raise NotImplementedError

    def setup_model(self, config):
        """"Not implemented."""
        raise NotImplementedError

    def extract_features(self, examples, config, resources):
        """Extracts all features from a list of MindMeld examples. Processes the data and returns the
        features in the format that is expected as an input to fit(). Note that the MindMeld config
        and resources are passed in each time to make the underlying model implementation stateless.

        Args:
            examples (list of mindmeld.core.Query): A list of queries to extract features for
            config (ModelConfig): The ModelConfig which may contain information used for feature
                                  extraction
            resources (dict): Resources which may be used for this model's feature extraction

        Returns:
            (tuple): tuple containing:

                * (list of feature vectors): X
                * (list of labels): y
                * (list of groups): A list of groups to be used for splitting with \
                    sklearn GridSearchCV
        """
        raise NotImplementedError

    def extract_and_predict(self, examples, config, resources):
        """Does both feature extraction and prediction. Often necessary for sequence models when the
        prediction of the previous example is used as a feature for the next example. If this is
        not the case, extract is simply called before predict here. Note that the MindMeld config
        and resources are passed in each time to make the underlying model implementation stateless.

        Args:
            examples (list of mindmeld.core.Query): A list of queries to extract features for and
                                                       predict
            config (ModelConfig): The ModelConfig which may contain information used for feature
                                  extraction
            resources (dict): Resources which may be used for this model's feature extraction

        Returns:
            (list of classification labels): A list of predicted labels (in encoded format)
        """
        X, _, _ = self.extract_features(examples, config, resources)
        y = self.predict(X)
        return y

    def predict_proba(self, examples, config, resources):
        """
        Args:
            examples (list of mindmeld.core.Query): A list of queries to extract features for and
                                                       predict
            config (ModelConfig): The ModelConfig which may contain information used for feature
                                  extraction
            resources (dict): Resources which may be used for this model's feature extraction

        Returns:
            (list of lists): A list of predicted labels (in encoded format) and confidence scores
        """
        X, _, _ = self.extract_features(examples, config, resources)
        return self._predict_proba(X)

    @staticmethod
    def _predict_proba(X):
        del X
        pass

    @property
    def is_serializable(self):
        # The default is True since < MM 3.2.0 models are serializable by default
        return True

    @staticmethod
    def dump(model_path):
        """
        Dumps the model to memory. This is a no-op since we do not
        have to do anything special to dump default serializable models
        for SKLearn.

        Args:
            model_path (str): The path to dump the model to
        """
        return model_path

    @staticmethod
    def unload():
        pass

    @staticmethod
    def load(model_path):
        """
        Load the model state to memory. This is a no-op since we do not
        have to do anything special to load default serializable models
        for SKLearn.

        Args:
            model_path (str): The path to dump the model to
        """
        del model_path
        pass
