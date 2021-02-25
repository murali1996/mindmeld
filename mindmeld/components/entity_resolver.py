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
This module contains the entity resolver component of the MindMeld natural language processor.
"""
import copy
import hashlib
import logging
import os
import pickle
from abc import ABC, abstractmethod

from elasticsearch.exceptions import ConnectionError as EsConnectionError
from elasticsearch.exceptions import ElasticsearchException, TransportError

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .. import path
from ..core import Entity
from ..exceptions import EntityResolverConnectionError, EntityResolverError
from ._config import (
    DEFAULT_ES_SYNONYM_MAPPING,
    PHONETIC_ES_SYNONYM_MAPPING,
    get_app_namespace,
    get_classifier_config,
)
from ._elasticsearch_helpers import (
    INDEX_TYPE_KB,
    INDEX_TYPE_SYNONYM,
    DOC_TYPE,
    create_es_client,
    delete_index,
    does_index_exist,
    get_field_names,
    get_scoped_index_name,
    load_index,
    resolve_es_config_for_version,
)

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Transformer, Pooling
    import torch
    from sentence_transformers.util import batch_to_device
    from tqdm.autonotebook import tqdm, trange

    sbert_available = True
except ImportError:
    sbert_available = False

logger = logging.getLogger(__name__)


class EntityResolver:
    """An entity resolver is used to resolve entities in a given query to their canonical values
    (usually linked to specific entries in a knowledge base).
    """

    @classmethod
    def validate_resolver_name(cls, name):
        if name not in ENTITY_RESOLVER_MODEL_TYPES:
            msg = "Expected 'model_type' in ENTITY_RESOLVER_CONFIG among {!r}"
            raise Exception(msg.format(ENTITY_RESOLVER_MODEL_TYPES))
        if name == "sbert_cosine_similarity" and not sbert_available:
            raise ImportError(
                "Must install the extra [bert] to use the built in embbedder for entity "
                "resolution. See https://www.mindmeld.com/docs/userguide/getting_started.html")

    def __new__(cls, app_path, resource_loader, entity_type, **kwargs):
        """Identifies appropriate entity resolver based on input config and
        initializes it.

        Args:
            app_path (str): The application path.
            resource_loader (ResourceLoader): An object which can load resources for the resolver.
            entity_type (str): The entity type associated with this entity resolver.
            er_config (dict): A classifier config
            es_host (str): The Elasticsearch host server.
            es_client (Elasticsearch): The Elasticsearch client.
        """
        er_config = (
            kwargs.pop("er_config", None) or
            get_classifier_config("entity_resolution", app_path=app_path)
        )
        name = er_config.get("model_type", None)
        cls.validate_resolver_name(name)
        return ENTITY_RESOLVER_MODEL_MAPPINGS.get(name)(
            app_path, resource_loader, entity_type, er_config, **kwargs
        )

    def fit(self, clean):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def predict(self, entity):
        raise NotImplementedError


class EntityResolverBase(ABC):
    """
    Base class for Entity Resolvers
    """

    def __init__(self, app_path, resource_loader, entity_type, er_config, **kwargs):
        """Initializes an entity resolver"""
        self.app_path = app_path
        self.resource_loader = resource_loader
        self.type = entity_type
        self.er_config = er_config
        self.kwargs = kwargs

        self._app_namespace = get_app_namespace(self.app_path)
        self._is_system_entity = Entity.is_system_entity(self.type)
        self.name = self.er_config.get("model_type")
        self.dirty = False  # bool, True if exists any unsaved generated data that can be saved
        self.ready = False  # bool, True if the model is fit by calling .fit()

        if self._is_system_entity:
            canonical_entities = []
        else:
            canonical_entities = self.resource_loader.get_entity_map(self.type).get(
                "entities", []
            )
        self._no_canonical_entity_map = len(canonical_entities) == 0

        if self._use_double_metaphone:
            self._invoke_double_metaphone_usage()

    @property
    def _use_double_metaphone(self):
        return "double_metaphone" in self.er_config.get("phonetic_match_types", [])

    def _invoke_double_metaphone_usage(self):
        """
        By default, resolvers are assumed to not support double metaphone usage
        If supported, override this method definition in the derived class
        (eg. see EntityResolverUsingElasticSearch)
        """
        logger.warning(
            "%r not configured to use double_metaphone",
            self.name
        )
        raise NotImplementedError

    def cache_path(self, tail_name=""):
        name = self.name + "_" + tail_name if tail_name else self.name
        return path.get_entity_resolver_cache_file_path(
            self.app_path, self.type, name
        )

    @abstractmethod
    def _fit(self, clean):
        raise NotImplementedError

    def fit(self, clean=False):
        """Fits the resolver model, if required

        Args:
            clean (bool, optional): If ``True``, deletes and recreates the index from scratch
                                    with synonyms in the mapping.json.
        """

        if self.ready:
            return

        if self._no_canonical_entity_map:
            return

        self._fit(clean)
        self.ready = True

    @abstractmethod
    def _predict(self, entity):
        raise NotImplementedError

    def predict(self, entity):
        """Predicts the resolved value(s) for the given entity using the loaded entity map or the
        trained entity resolution model.

        Args:
            entity (Entity, tuple): An entity found in an input query, or a list of n-best entity \
                objects.

        Returns:
            (list): The top 20 resolved values for the provided entity.
        """
        if isinstance(entity, (list, tuple)):
            top_entity = entity[0]
            entity = tuple(entity)
        else:
            top_entity = entity
            entity = tuple([entity])

        if self._is_system_entity:
            # system entities are already resolved
            return [top_entity.value]

        if self._no_canonical_entity_map:
            return []

        return self._predict(entity)

    @abstractmethod
    def _load(self):
        raise NotImplementedError

    def load(self):
        """If available, loads embeddings of synonyms that are previously dumped
        """
        self._load()

    def _dump(self):
        raise NotImplementedError

    def __repr__(self):
        msg = "<{} {!r} ready: {!r}, dirty: {!r}>"
        return msg.format(self.__class__.__name__, self.name, self.ready, self.dirty)


class EntityResolverUsingElasticSearch(EntityResolverBase):
    """
    Resolver class based on Elastic Search
    """

    # prefix for Elasticsearch indices used to store synonyms for entity resolution
    ES_SYNONYM_INDEX_PREFIX = "synonym"
    """The prefix of the ES index."""

    def __init__(self, app_path, resource_loader, entity_type, er_config, **kwargs):
        super().__init__(app_path, resource_loader, entity_type, er_config, **kwargs)
        self._es_host = self.kwargs.get("es_host", None)
        self._es_config = {"client": self.kwargs.get("es_client", None), "pid": os.getpid()}

    def _invoke_double_metaphone_usage(self):
        pass

    @property
    def _es_index_name(self):
        return EntityResolverUsingElasticSearch.ES_SYNONYM_INDEX_PREFIX + "_" + self.type

    @property
    def _es_client(self):
        # Lazily connect to Elasticsearch.  Make sure each subprocess gets it's own connection
        if self._es_config["client"] is None or self._es_config["pid"] != os.getpid():
            self._es_config = {"pid": os.getpid(), "client": create_es_client()}
        return self._es_config["client"]

    @classmethod
    def ingest_synonym(
        cls,
        app_namespace,
        index_name,
        index_type=INDEX_TYPE_SYNONYM,
        field_name=None,
        data=None,
        es_host=None,
        es_client=None,
        use_double_metaphone=False,
    ):
        """Loads synonym documents from the mapping.json data into the
        specified index. If an index with the specified name doesn't exist, a
        new index with that name will be created.

        Args:
            app_namespace (str): The namespace of the app. Used to prevent
                collisions between the indices of this app and those of other
                apps.
            index_name (str): The name of the new index to be created.
            index_type (str): specify whether to import to synonym index or
                knowledge base object index. INDEX_TYPE_SYNONYM is the default
                which indicates the synonyms to be imported to synonym index,
                while INDEX_TYPE_KB indicates that the synonyms should be
                imported into existing knowledge base index.
            field_name (str): specify name of the knowledge base field that the
                synonym list corresponds to when index_type is
                INDEX_TYPE_SYNONYM.
            data (list): A list of documents to be loaded into the index.
            es_host (str): The Elasticsearch host server.
            es_client (Elasticsearch): The Elasticsearch client.
            use_double_metaphone (bool): Whether to use the phonetic mapping or not.
        """
        data = data or []

        def _action_generator(docs):

            for doc in docs:
                action = {}

                # id
                if doc.get("id"):
                    action["_id"] = doc["id"]
                else:
                    # generate hash from canonical name as ID
                    action["_id"] = hashlib.sha256(
                        doc.get("cname").encode("utf-8")
                    ).hexdigest()

                # synonym whitelist
                whitelist = doc["whitelist"]
                syn_list = []
                syn_list.append({"name": doc["cname"]})
                for syn in whitelist:
                    syn_list.append({"name": syn})

                # If index type is INDEX_TYPE_KB  we import the synonym into knowledge base object
                # index by updating the knowledge base object with additional synonym whitelist
                # field. Otherwise, by default we import to synonym index in ES.
                if index_type == INDEX_TYPE_KB and field_name:
                    syn_field = field_name + "$whitelist"
                    action["_op_type"] = "update"
                    action["doc"] = {syn_field: syn_list}
                else:
                    action.update(doc)
                    action["whitelist"] = syn_list

                yield action

        mapping = (
            PHONETIC_ES_SYNONYM_MAPPING
            if use_double_metaphone
            else DEFAULT_ES_SYNONYM_MAPPING
        )
        es_client = es_client or create_es_client(es_host)
        mapping = resolve_es_config_for_version(mapping, es_client)
        load_index(
            app_namespace,
            index_name,
            _action_generator(data),
            len(data),
            mapping,
            DOC_TYPE,
            es_host,
            es_client,
        )

    def _fit(self, clean):
        """Loads an entity mapping file to Elasticsearch for text relevance based entity resolution.

        In addition, the synonyms in entity mapping are imported to knowledge base indexes if the
        corresponding knowledge base object index and field name are specified for the entity type.
        The synonym info is then used by Question Answerer for text relevance matches.

        Args:
            clean (bool): If ``True``, deletes and recreates the index from scratch instead of
                          updating the existing index with synonyms in the mapping.json.
        """
        if clean:
            delete_index(
                self._app_namespace, self._es_index_name, self._es_host, self._es_client
            )

        entity_map = self.resource_loader.get_entity_map(self.type)

        # list of canonical entities and their synonyms
        entities = entity_map.get("entities", [])

        # create synonym index and import synonyms
        logger.info("Importing synonym data to synonym index '%s'", self._es_index_name)
        EntityResolverUsingElasticSearch.ingest_synonym(
            app_namespace=self._app_namespace,
            index_name=self._es_index_name,
            data=entities,
            es_host=self._es_host,
            es_client=self._es_client,
            use_double_metaphone=self._use_double_metaphone,
        )

        # It's supported to specify the KB object type and field name that the NLP entity type
        # corresponds to in the mapping.json file. In this case the synonym whitelist is also
        # imported to KB object index and the synonym info will be used when using Question Answerer
        # for text relevance matches.
        kb_index = entity_map.get("kb_index_name")
        kb_field = entity_map.get("kb_field_name")

        # if KB index and field name is specified then also import synonyms into KB object index.
        if kb_index and kb_field:
            # validate the KB index and field are valid.
            # TODO: this validation can probably be in some other places like resource loader.
            if not does_index_exist(
                self._app_namespace, kb_index, self._es_host, self._es_client
            ):
                raise ValueError(
                    "Cannot import synonym data to knowledge base. The knowledge base "
                    "index name '{}' is not valid.".format(kb_index)
                )
            if kb_field not in get_field_names(
                self._app_namespace, kb_index, self._es_host, self._es_client
            ):
                raise ValueError(
                    "Cannot import synonym data to knowledge base. The knowledge base "
                    "field name '{}' is not valid.".format(kb_field)
                )
            if entities and not entities[0].get("id"):
                raise ValueError(
                    "Knowledge base index and field cannot be specified for entities "
                    "without ID."
                )
            logger.info("Importing synonym data to knowledge base index '%s'", kb_index)
            EntityResolverUsingElasticSearch.ingest_synonym(
                app_namespace=self._app_namespace,
                index_name=kb_index,
                index_type="kb",
                field_name=kb_field,
                data=entities,
                es_host=self._es_host,
                es_client=self._es_client,
                use_double_metaphone=self._use_double_metaphone,
            )

    def _predict(self, entity):
        """Predicts the resolved value(s) for the given entity using the loaded entity map or the
        trained entity resolution model.

        Args:
            entity (Entity, tuple): An entity found in an input query, or a list of n-best entity \
                objects.

        Returns:
            (list): The top 20 resolved values for the provided entity.
        """

        top_entity = entity[0]

        weight_factors = [1 - float(i) / len(entity) for i in range(len(entity))]

        def _construct_match_query(entity, weight=1):
            return [
                {
                    "match": {
                        "cname.normalized_keyword": {
                            "query": entity.text,
                            "boost": 10 * weight,
                        }
                    }
                },
                {"match": {"cname.raw": {"query": entity.text, "boost": 10 * weight}}},
                {
                    "match": {
                        "cname.char_ngram": {"query": entity.text, "boost": weight}
                    }
                },
            ]

        def _construct_nbest_match_query(entity, weight=1):
            return [
                {
                    "match": {
                        "cname.normalized_keyword": {
                            "query": entity.text,
                            "boost": weight,
                        }
                    }
                }
            ]

        def _construct_phonetic_match_query(entity, weight=1):
            return [
                {
                    "match": {
                        "cname.double_metaphone": {
                            "query": entity.text,
                            "boost": 2 * weight,
                        }
                    }
                }
            ]

        def _construct_whitelist_query(entity, weight=1, use_phons=False):
            query = {
                "nested": {
                    "path": "whitelist",
                    "score_mode": "max",
                    "query": {
                        "bool": {
                            "should": [
                                {
                                    "match": {
                                        "whitelist.name.normalized_keyword": {
                                            "query": entity.text,
                                            "boost": 10 * weight,
                                        }
                                    }
                                },
                                {
                                    "match": {
                                        "whitelist.name": {
                                            "query": entity.text,
                                            "boost": weight,
                                        }
                                    }
                                },
                                {
                                    "match": {
                                        "whitelist.name.char_ngram": {
                                            "query": entity.text,
                                            "boost": weight,
                                        }
                                    }
                                },
                            ]
                        }
                    },
                    "inner_hits": {},
                }
            }

            if use_phons:
                query["nested"]["query"]["bool"]["should"].append(
                    {
                        "match": {
                            "whitelist.double_metaphone": {
                                "query": entity.text,
                                "boost": 3 * weight,
                            }
                        }
                    }
                )

            return query

        text_relevance_query = {
            "query": {
                "function_score": {
                    "query": {"bool": {"should": []}},
                    "field_value_factor": {
                        "field": "sort_factor",
                        "modifier": "log1p",
                        "factor": 10,
                        "missing": 0,
                    },
                    "boost_mode": "sum",
                    "score_mode": "sum",
                }
            }
        }

        match_query = []
        top_transcript = True
        for e, weight in zip(entity, weight_factors):
            if top_transcript:
                match_query.extend(_construct_match_query(e, weight))
                top_transcript = False
            else:
                match_query.extend(_construct_nbest_match_query(e, weight))
            if self._use_double_metaphone:
                match_query.extend(_construct_phonetic_match_query(e, weight))
        text_relevance_query["query"]["function_score"]["query"]["bool"][
            "should"
        ].append({"bool": {"should": match_query}})

        whitelist_query = _construct_whitelist_query(
            top_entity, use_phons=self._use_double_metaphone
        )
        text_relevance_query["query"]["function_score"]["query"]["bool"][
            "should"
        ].append(whitelist_query)

        try:
            index = get_scoped_index_name(self._app_namespace, self._es_index_name)
            response = self._es_client.search(index=index, body=text_relevance_query)
        except EsConnectionError as ex:
            logger.error(
                "Unable to connect to Elasticsearch: %s details: %s", ex.error, ex.info
            )
            raise EntityResolverConnectionError(es_host=self._es_client.transport.hosts) from ex
        except TransportError as ex:
            logger.error(
                "Unexpected error occurred when sending requests to Elasticsearch: %s "
                "Status code: %s details: %s",
                ex.error,
                ex.status_code,
                ex.info,
            )
            raise EntityResolverError(
                "Unexpected error occurred when sending requests to "
                "Elasticsearch: {} Status code: {} details: "
                "{}".format(ex.error, ex.status_code, ex.info)
            ) from ex
        except ElasticsearchException as ex:
            raise EntityResolverError from ex
        else:
            hits = response["hits"]["hits"]

            results = []
            for hit in hits:
                if self._use_double_metaphone and len(entity) > 1:
                    if hit["_score"] < 0.5 * len(entity):
                        continue

                top_synonym = None
                synonym_hits = hit["inner_hits"]["whitelist"]["hits"]["hits"]
                if synonym_hits:
                    top_synonym = synonym_hits[0]["_source"]["name"]
                result = {
                    "cname": hit["_source"]["cname"],
                    "score": hit["_score"],
                    "top_synonym": top_synonym,
                }

                if hit["_source"].get("id"):
                    result["id"] = hit["_source"].get("id")

                if hit["_source"].get("sort_factor"):
                    result["sort_factor"] = hit["_source"].get("sort_factor")

                results.append(result)

            return results[0:20]

    def _load(self):
        """Loads the trained entity resolution model from disk."""
        try:
            scoped_index_name = get_scoped_index_name(
                self._app_namespace, self._es_index_name
            )
            if not self._es_client.indices.exists(index=scoped_index_name):
                self.fit()
        except EsConnectionError as e:
            logger.error(
                "Unable to connect to Elasticsearch: %s details: %s", e.error, e.info
            )
            raise EntityResolverConnectionError(es_host=self._es_client.transport.hosts) from e
        except TransportError as e:
            logger.error(
                "Unexpected error occurred when sending requests to Elasticsearch: %s "
                "Status code: %s details: %s",
                e.error,
                e.status_code,
                e.info,
            )
            raise EntityResolverError from e
        except ElasticsearchException as e:
            raise EntityResolverError from e


class EntityResolverUsingExactMatch(EntityResolverBase):
    """
    Resolver class based on exact matching
    """

    def __init__(self, app_path, resource_loader, entity_type, er_config, **kwargs):
        super().__init__(app_path, resource_loader, entity_type, er_config, **kwargs)
        self._normalizer = self.resource_loader.query_factory.normalize
        self._exact_match_mapping = None

    @staticmethod
    def _process_entity_map(entity_type, entity_map, normalizer):
        """Loads in the mapping.json file and stores the synonym mappings in a item_map and a
        synonym_map for exact match entity resolution when Elasticsearch is unavailable

        Args:
            entity_type: The entity type associated with this entity resolver
            entity_map: The loaded mapping.json file for the given entity type
            normalizer: The normalizer to use
        """
        item_map = {}
        syn_map = {}
        seen_ids = []
        for item in entity_map.get("entities"):
            cname = item["cname"]
            item_id = item.get("id")
            if cname in item_map:
                msg = "Canonical name %s specified in %s entity map multiple times"
                logger.debug(msg, cname, entity_type)
            if item_id:
                if item_id in seen_ids:
                    msg = "Item id {!r} specified in {!r} entity map multiple times"
                    raise ValueError(msg.format(item_id, entity_type))
                seen_ids.append(item_id)

            aliases = [cname] + item.pop("whitelist", [])
            items_for_cname = item_map.get(cname, [])
            items_for_cname.append(item)
            item_map[cname] = items_for_cname
            for alias in aliases:
                norm_alias = normalizer(alias)
                if norm_alias in syn_map:
                    msg = "Synonym %s specified in %s entity map multiple times"
                    logger.debug(msg, cname, entity_type)
                cnames_for_syn = syn_map.get(norm_alias, [])
                cnames_for_syn.append(cname)
                syn_map[norm_alias] = list(set(cnames_for_syn))

        return {"items": item_map, "synonyms": syn_map}

    def _fit(self, clean):
        """Loads an entity mapping file to resolve entities using exact match.
        """
        if clean:
            logger.info(
                "clean=True ignored while fitting exact_match algo for entity resolution"
            )

        entity_map = self.resource_loader.get_entity_map(self.type)
        self._exact_match_mapping = self._process_entity_map(
            self.type, entity_map, self._normalizer
        )

    def _predict(self, entity):
        """Looks for exact name in the synonyms data
        """

        entity = entity[0]  # top_entity

        normed = self._normalizer(entity.text)
        try:
            cnames = self._exact_match_mapping["synonyms"][normed]
        except (KeyError, TypeError):
            logger.warning(
                "Failed to resolve entity %r for type %r", entity.text, entity.type
            )
            return None

        if len(cnames) > 1:
            logger.info(
                "Multiple possible canonical names for %r entity for type %r",
                entity.text,
                entity.type,
            )

        values = []
        for cname in cnames:
            for item in self._exact_match_mapping["items"][cname]:
                item_value = copy.copy(item)
                item_value.pop("whitelist", None)
                values.append(item_value)

        return values

    def _load(self):
        self.fit()


class EntityResolverUsingSentenceBertEmbedder(EntityResolverBase):
    """
    Resolver class for bert models as described here:
    https://github.com/UKPLab/sentence-transformers
    """

    def __init__(self, app_path, resource_loader, entity_type, er_config, **kwargs):
        super().__init__(app_path, resource_loader, entity_type, er_config, **kwargs)
        self._exact_match_mapping = None
        self._preloaded_mappings_embs = {}
        self._sbert_model_pretrained_name_or_abspath = (
            self.er_config.get("model_settings", {})
                .get("pretrained_name_or_abspath", "bert-base-nli-mean-tokens")
        )
        self._sbert_model = None
        self._augment_lower_case = False

        # TODO: _lazy_resolution is set to a default value, can be modified to be an input
        self._lazy_resolution = False
        if not self._lazy_resolution:
            msg = "sentence-bert embeddings are cached for entity_type: {%s} " \
                  "for fast entity resolution; can possibly consume more disk space"
            logger.warning(msg, self.type)

    @property
    def pretrained_name(self):
        return os.path.split(self._sbert_model_pretrained_name_or_abspath)[-1]

    def _encode(self, phrases):
        """Encodes input text(s) into embeddings, one vector for each phrase

        Args:
            phrases (str, list[str]): textual inputs that are to be encoded using sentence \
                                        transformers' model

        Returns:
            list[np.array]: one numpy array of embeddings for each phrase,
                            if ``phrases`` is ``str``, a list of one numpy aray is returned
        """

        if not phrases:
            return []

        if isinstance(phrases, str):
            phrases = [phrases]

        if not isinstance(phrases, (str, list)):
            raise TypeError(f"argument phrases must be of type str or list, not {type(phrases)}")

        # batch_size (int):
        #   The maximum size of each batch while encoding using on a deep embedder like BERT
        _batch_size = (
            self.er_config.get("model_settings", {})
                .get("batch_size", 16)
        )
        show_progress = len(phrases) > 1
        convert_to_numpy = True

        normalize_embs = False
        if normalize_embs:
            encode_func = self._encode_normalized
        else:
            encode_func = self._sbert_model.encode

        concat_embs = True
        if concat_embs:
            encode_func = self._encode_concatenated
        else:
            encode_func = self._sbert_model.encode

        return encode_func(phrases, batch_size=_batch_size, convert_to_numpy=convert_to_numpy,
                           show_progress_bar=show_progress)

    @staticmethod
    def _text_length(text):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).

        Union[List[int], List[List[int]]]
        """
        if isinstance(text, dict):
            return len(next(iter(text.values())))
        elif len(text) == 0 or isinstance(text[0], int):
            return len(text)
        else:
            return sum([len(t) for t in text])

    def _encode_concatenated(self, sentences,
                             batch_size: int = 32,
                             show_progress_bar: bool = None,
                             output_value: str = 'sentence_embedding',
                             convert_to_numpy: bool = True,
                             convert_to_tensor: bool = False,
                             is_pretokenized: bool = False,
                             device: str = None,
                             num_workers: int = 0):
        """
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings.
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param is_pretokenized: DEPRECATED - No longer used, will be removed in the future
        :param device: Which torch.device to use for the computation
        :param num_workers: DEPRECATED - No longer used, will be removed in the future

        :return: (Union[List[Tensor], ndarray, Tensor])
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """

        """
        in sentence-transformers, in Transformers.py
            added ```config.output_hidden_states = True```
        """

        self.transformer_model.eval()
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel()==logging.INFO or logger.getEffectiveLevel()==logging.DEBUG)

        if convert_to_tensor:
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str): #Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transformer_model.to(device)
        self.pooling_model.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            features = self.transformer_model.tokenize(sentences_batch)
            features = batch_to_device(features, device)

            with torch.no_grad():
                _out_features_transformer = self.transformer_model.forward(features)  # from transformer model
                _all_layer_embeddings = _out_features_transformer["all_layer_embeddings"]
                _token_embeddings = torch.cat(_all_layer_embeddings[-4:], dim=-1)
                # _token_embeddings = _out_features_transformer["token_embeddings"]
                # _norm_token_embeddings = torch.linalg.norm(_token_embeddings, dim=2, keepdim=True)
                # _token_embeddings = _token_embeddings.div(_norm_token_embeddings)
                _out_features_transformer.update({"token_embeddings": _token_embeddings})
                out_features = self.pooling_model.forward(_out_features_transformer)

                embeddings = out_features[output_value]

                if output_value == 'token_embeddings':
                    #Set token embeddings to 0 for padding tokens
                    input_mask = out_features['attention_mask']
                    input_mask_expanded = input_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    embeddings = embeddings * input_mask_expanded

                embeddings = embeddings.detach()

                # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                if convert_to_numpy:
                    embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def _encode_normalized(self, sentences,
                           batch_size: int = 32,
                           show_progress_bar: bool = None,
                           output_value: str = 'sentence_embedding',
                           convert_to_numpy: bool = True,
                           convert_to_tensor: bool = False,
                           is_pretokenized: bool = False,
                           device: str = None,
                           num_workers: int = 0):
        """
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings.
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param is_pretokenized: DEPRECATED - No longer used, will be removed in the future
        :param device: Which torch.device to use for the computation
        :param num_workers: DEPRECATED - No longer used, will be removed in the future

        :return: (Union[List[Tensor], ndarray, Tensor])
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """

        self.transformer_model.eval()
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel()==logging.INFO or logger.getEffectiveLevel()==logging.DEBUG)

        if convert_to_tensor:
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str): #Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transformer_model.to(device)
        self.pooling_model.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            features = self.transformer_model.tokenize(sentences_batch)
            features = batch_to_device(features, device)

            with torch.no_grad():
                _out_features_transformer = self.transformer_model.forward(features)  # from transformer model
                _token_embeddings = _out_features_transformer["token_embeddings"]
                _norm_token_embeddings = torch.linalg.norm(_token_embeddings, dim=2, keepdim=True)
                _token_embeddings = _token_embeddings.div(_norm_token_embeddings)
                _out_features_transformer.update({"token_embeddings": _token_embeddings})
                out_features = self.pooling_model.forward(_out_features_transformer)

                embeddings = out_features[output_value]

                if output_value == 'token_embeddings':
                    #Set token embeddings to 0 for padding tokens
                    input_mask = out_features['attention_mask']
                    input_mask_expanded = input_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    embeddings = embeddings * input_mask_expanded

                embeddings = embeddings.detach()

                # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                if convert_to_numpy:
                    embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    @staticmethod
    def _compute_cosine_similarity(syn_embs, entity_emb, return_as_dict=False):
        """Uses cosine similarity metric on synonym embeddings to sort most relevant ones
            for entity resolution

        Args:
            syn_embs (dict): a dict of synonym and its corresponding embedding from bert
            entity_emb (np.array): embedding of the input entity text, an array of size 1
        Returns:
            Union[dict, list[tuple]]: if return_as_dict, returns a dictionary of synonyms and their
                                        scores, else a list of sorted synonym names, paired with
                                        their similarity scores (descending)
        """

        entity_emb = entity_emb.reshape(1, -1)
        synonyms, synonyms_encodings = zip(*syn_embs.items())
        similarity_scores = cosine_similarity(np.array(synonyms_encodings), entity_emb).reshape(-1)
        similarity_scores = np.around(similarity_scores, decimals=2)

        if return_as_dict:
            return dict(zip(synonyms, similarity_scores))

        # results in descending scores
        return sorted(list(zip(synonyms, similarity_scores)), key=lambda x: x[1], reverse=True)

    def _process_entity_map(self, entity_type, entity_map):
        """Loads in the mapping.json file and stores the synonym mappings in a item_map and a
        synonym_map for exact match entity resolution when Elasticsearch is unavailable

        Args:
            entity_type: The entity type associated with this entity resolver
            entity_map: The loaded mapping.json file for the given entity type
        """
        item_map = {}
        syn_map = {}
        seen_ids = []
        for item in entity_map.get("entities"):
            cname = item["cname"]
            item_id = item.get("id")
            if cname in item_map:
                msg = "Canonical name %s specified in %s entity map multiple times"
                logger.debug(msg, cname, entity_type)
            if item_id:
                if item_id in seen_ids:
                    msg = "Item id {!r} specified in {!r} entity map multiple times"
                    raise ValueError(msg.format(item_id, entity_type))
                seen_ids.append(item_id)

            aliases = [cname] + item.pop("whitelist", [])
            items_for_cname = item_map.get(cname, [])
            items_for_cname.append(item)
            item_map[cname] = items_for_cname
            for norm_alias in aliases:
                if norm_alias in syn_map:
                    msg = "Synonym %s specified in %s entity map multiple times"
                    logger.debug(msg, cname, entity_type)
                cnames_for_syn = syn_map.get(norm_alias, [])
                cnames_for_syn.append(cname)
                syn_map[norm_alias] = list(set(cnames_for_syn))

        # extend synonyms map by adding keys which are lowercases of the existing keys
        if self._augment_lower_case:
            msg = "Adding lowercased whitelist and cnames to list of possible synonyms"
            logger.info(msg)
            initial_num_syns = len(syn_map)
            aug_syn_map = {}
            for alias, alias_map in syn_map.items():
                alias_lower = alias.lower()
                if alias_lower not in syn_map:
                    aug_syn_map.update({alias_lower: alias_map})
            syn_map.update(aug_syn_map)
            final_num_syns = len(syn_map)
            msg = "Added %d additional synonyms by lower-casing. Upped from %d to %d"
            logger.info(msg, final_num_syns - initial_num_syns, initial_num_syns, final_num_syns)

        return {"items": item_map, "synonyms": syn_map}

    def _fit(self, clean):
        """
        Fits the resolver model

        Args:
            clean (bool): If ``True``, deletes existing dump of synonym embeddings file
        """

        _bert_output_type = (
            self.er_config.get("model_settings", {})
                .get("bert_output_type", "mean")
        )

        # load model
        try:
            self.transformer_model = Transformer(self._sbert_model_pretrained_name_or_abspath)
            self.pooling_model = Pooling(self.transformer_model.get_word_embedding_dimension(),
                                    pooling_mode_cls_token=_bert_output_type == "cls",
                                    pooling_mode_max_tokens=False,
                                    pooling_mode_mean_tokens=_bert_output_type == "mean",
                                    pooling_mode_mean_sqrt_len_tokens=False)
            modules = [self.transformer_model, self.pooling_model]
            self._sbert_model = SentenceTransformer(modules=modules)
        except OSError:
            logger.error("Could not initialize the model name through huggingface models; "
                         "Checking an alternate name - %s - in huggingface models",
                         "sentence-transformers/" + self._sbert_model_pretrained_name_or_abspath)
            try:
                _sbert_model_pretrained_name_or_abspath = \
                    "sentence-transformers/" + self._sbert_model_pretrained_name_or_abspath
                self.transformer_model = Transformer(_sbert_model_pretrained_name_or_abspath)
                self.pooling_model = Pooling(self.transformer_model.get_word_embedding_dimension(),
                                        pooling_mode_cls_token=_bert_output_type == "cls",
                                        pooling_mode_max_tokens=False,
                                        pooling_mode_mean_tokens=_bert_output_type == "mean",
                                        pooling_mode_mean_sqrt_len_tokens=False)
                modules = [self.transformer_model, self.pooling_model]
                self._sbert_model = SentenceTransformer(modules=modules)
            except OSError:
                logger.error("Could not initialize the model name through huggingface models; "
                             "Resorting to model names in sbert.net. "
                             "Input 'bert_output_type', if provided is thus ignored")
                self._sbert_model = SentenceTransformer(self._sbert_model_pretrained_name_or_abspath)

        # load mappings.json data
        entity_map = self.resource_loader.get_entity_map(self.type)
        self._exact_match_mapping = self._process_entity_map(
            self.type, entity_map
        )

        # load embeddings for this data
        cache_path = self.cache_path(self.pretrained_name)
        if clean and os.path.exists(cache_path):
            os.remove(cache_path)
        if not self._lazy_resolution and os.path.exists(cache_path):
            self._load()
            self.dirty = False
        else:
            synonyms = [*self._exact_match_mapping["synonyms"]]
            synonyms_encodings = self._encode(synonyms)
            self._preloaded_mappings_embs = dict(zip(synonyms, synonyms_encodings))
            self.dirty = True

        if self.dirty and not self._lazy_resolution:
            self._dump()

    def _predict(self, entity):
        """Predicts the resolved value(s) for the given entity using cosine similarity.

        Args:
            entity (Entity, tuple): An entity found in an input query, or a list of n-best entity \
                objects.

        Returns:
            (list): The top 20 resolved values for the provided entity.
        """

        syn_embs = self._preloaded_mappings_embs
        entity = entity[0]  # top_entity
        entity_emb = self._encode(entity.text)[0]

        try:
            sorted_items = self._compute_cosine_similarity(syn_embs, entity_emb)
            values = []
            for synonym, score in sorted_items:
                cnames = self._exact_match_mapping["synonyms"][synonym]
                for cname in cnames:
                    for item in self._exact_match_mapping["items"][cname]:
                        item_value = copy.copy(item)
                        item_value.pop("whitelist", None)
                        item_value.update({"score": score})
                        values.append(item_value)
        except KeyError:
            logger.warning(
                "Failed to resolve entity %r for type %r; "
                "set 'clean=True' for computing embeddings of newly added items in mappings.json",
                entity.text, entity.type
            )
            return None
        except TypeError:
            logger.warning(
                "Failed to resolve entity %r for type %r", entity.text, entity.type
            )
            return None

        # combine_scores_type = (
        #     self.er_config.get("model_settings", {}).get(
        #         "combine_scores_type", "none")
        # )
        # try:
        #     if combine_scores_type == "none":
        #         sorted_items = self._compute_cosine_similarity(syn_embs, entity_emb)
        #         values = []
        #         for synonym, score in sorted_items:
        #             cnames = self._exact_match_mapping["synonyms"][synonym]
        #             for cname in cnames:
        #                 for item in self._exact_match_mapping["items"][cname]:
        #                     item_value = copy.copy(item)
        #                     item_value.pop("whitelist", None)
        #                     item_value.update({"score": score})
        #                     values.append(item_value)
        #     else:
        #         scored_items = self._compute_cosine_similarity(syn_embs, entity_emb,
        #                                                        return_as_dict=True)
        #         # TODO: `self.resource_loader.get_entity_map` is loaded multiple times throughout
        #         cname_groups = self.resource_loader.get_entity_map(self.type).get("entities")
        #         values = []
        #         for group in cname_groups:
        #             group_syms = group.get("whitelist", []) + [group["cname"]]
        #             scores = [scored_items[sym] for sym in group_syms]
        #             group_clone = copy.copy(group)
        #             group_clone.pop("whitelist", None)
        #             if combine_scores_type == "mean":
        #                 group_clone.update({"score": float(sum(scores)) / len(scores)})
        #             elif combine_scores_type == "max":
        #                 group_clone.update({"score": max(scores)})
        #             values.append(group_clone)
        #         values = sorted(values, key=lambda x: x["score"], reverse=True)
        # except KeyError:
        #     logger.warning(
        #         "Failed to resolve entity %r for type %r; "
        #         "set 'clean=True' for computing embeddings of newly added items in mappings.json",
        #         entity.text, entity.type
        #     )
        #     return None
        # except TypeError:
        #     logger.warning(
        #         "Failed to resolve entity %r for type %r", entity.text, entity.type
        #     )
        #     return None

        # pool_syn_embs = (
        #     self.er_config.get("model_settings", {}).get(
        #         "pool_syn_embs", "none")
        # )
        # assert pool_syn_embs != "none"
        # cname_groups = self.resource_loader.get_entity_map(self.type).get("entities")
        # new_syn_embs = {}
        # for group in cname_groups:
        #     group_syms = group.get("whitelist", []) + [group["cname"]]
        #     group_emb = getattr(np, pool_syn_embs)([syn_embs[sym] for sym in group_syms], axis=0)
        #     new_syn_embs.update({group["cname"]: group_emb})
        # sorted_items = self._compute_cosine_similarity(new_syn_embs, entity_emb)
        # values = []
        # for synonym, score in sorted_items:
        #     cnames = self._exact_match_mapping["synonyms"][synonym]
        #     for cname in cnames:
        #         for item in self._exact_match_mapping["items"][cname]:
        #             item_value = copy.copy(item)
        #             item_value.pop("whitelist", None)
        #             item_value.update({"score": score})
        #             values.append(item_value)

        return values[0:20]

    def _load(self):
        """Loads embeddings for all synonyms, previously dumped into a .pkl file
        """
        cache_path = self.cache_path(self.pretrained_name)
        with open(cache_path, "rb") as fp:
            self._preloaded_mappings_embs = pickle.load(fp)

    def _dump(self):
        """Dumps embeddings of synonyms into a .pkl file when the .fit() method is called
        """
        cache_path = self.cache_path(self.pretrained_name)
        if self.dirty:
            folder = os.path.split(cache_path)[0]
            if folder and not os.path.exists(folder):
                os.makedirs(folder)
            with open(cache_path, "wb") as fp:
                pickle.dump(self._preloaded_mappings_embs, fp)


ENTITY_RESOLVER_MODEL_MAPPINGS = {
    "exact_match": EntityResolverUsingExactMatch,
    "text_relevance": EntityResolverUsingElasticSearch,
    "sbert_cosine_similarity": EntityResolverUsingSentenceBertEmbedder
}
ENTITY_RESOLVER_MODEL_TYPES = [*ENTITY_RESOLVER_MODEL_MAPPINGS]
