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

import copy
import logging

from ..helpers import ENABLE_STEMMING, get_feature_extractor

logger = logging.getLogger(__name__)


def extract_sequence_features(example, example_type, feature_config, resources):
    """Extracts feature dicts for each token in an example.

    Args:
        example (mindmeld.core.Query): a query
        example_type (str): The type of example
        feature_config (dict): The config for features
        resources (dict): Resources of this model

    Returns:
        (list of dict): features
    """
    feat_seq = []
    workspace_features = copy.deepcopy(feature_config)
    enable_stemming = workspace_features.pop(ENABLE_STEMMING, False)

    for name, kwargs in workspace_features.items():
        if callable(kwargs):
            # a feature extractor function was passed in directly
            feat_extractor = kwargs
        else:
            kwargs[ENABLE_STEMMING] = enable_stemming
            feat_extractor = get_feature_extractor(example_type, name)(**kwargs)

        update_feat_seq = feat_extractor(example, resources)
        if not feat_seq:
            feat_seq = update_feat_seq
        else:
            for idx, features in enumerate(update_feat_seq):
                feat_seq[idx].update(features)

    return feat_seq
