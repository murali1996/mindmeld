"""
This module contains the entity resolver component of the MindMeld natural language processor.
"""
import copy
import hashlib
import logging
import os

from elasticsearch.exceptions import ConnectionError as EsConnectionError
from elasticsearch.exceptions import ElasticsearchException, TransportError

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

from abc import ABC, abstractmethod

ENTITY_RESOLVER_MODEL_MAPPINGS = {
    "exact_match": EntityResolverUsingExactSearch,
    "text_relevance": EntityResolverUsingElasticSearch,
}
ENTITY_RESOLVER_MODEL_TYPES = [*ENTITY_RESOLVER_MODEL_MAPPINGS]

logger = logging.getLogger(__name__)


class EntityResolver:

    @classmethod
    def validate_resolver_name(name):
        if not name in ENTITY_RESOLVER_MODEL_TYPES:
            msg = "Expected 'model_type' in ENTITY_RESOLVER_CONFIG among {}"
            raise Exception(msg.format(ENTITY_RESOLVER_MODEL_TYPES))

    def __new__(cls, app_path, resource_loader, entity_type, **kwargs):
        er_config = get_classifier_config("entity_resolution", app_path=app_path)
        name = er_config.get("model_type", None)
        cls.validate_resolver_name(name)
        return ENTITY_RESOLVER_MODEL_MAPPINGS.get(name)(
            app_path, resource_loader, entity_type, er_config, **kwargs
        )


class EntityResolverBase(ABC):

    def __init__(self, app_path, resource_loader, entity_type, er_config, **kwargs):
        self._app_namespace = get_app_namespace(app_path)
        self._resource_loader = resource_loader
        self.type = entity_type
        self._is_system_entity = Entity.is_system_entity(self.type)
        self._er_config = er_config
        self.name = self._er_config.get("model_type")
        self.ready = False
        # TODO: save model when dirty is true
        self.dirty = False  # bool -- Whether the classifier has unsaved changes to its model.

        if self._is_system_entity:
            canonical_entities = []
        else:
            canonical_entities = self._resource_loader.get_entity_map(self.type).get(
                "entities", []
            )
            self.__validate_resolver_name(self.name)
        self._no_canonical_entity_map = len(canonical_entities) == 0

    @property
    def _use_double_metaphone(self):
        return "double_metaphone" in self._er_config.get("phonetic_match_types", [])

    @abstractmethod
    def _fit(self):
        raise NotImplementedError

    def fit(self, clean=False):

        if self.ready:
            return

        if self._no_canonical_entity_map:
            return

        self._fit(clean)
        self.ready = True

    @abstractmethod
    def _predict(self):
        raise NotImplementedError

    def predict(self, entity):

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
        self._load()

    # TODO: make this an abstractmethod and
    #   add dump functionalities in derived classes
    def _dump(self):
        raise NotImplementedError

    def __repr__(self):
        msg = "<{} {!r} ready: {!r}, dirty: {!r}>"
        return msg.format(self.__class__.__name__, self.name, self.ready, self.dirty)


class EntityResolverUsingElasticSearch(EntityResolverBase):

    # prefix for Elasticsearch indices used to store synonyms for entity resolution
    ES_SYNONYM_INDEX_PREFIX = "synonym"

    def __init__(self, app_path, resource_loader, entity_type, er_config, **kwargs):
        super(EntityResolverUsingElasticSearch, self).__init__(app_path, resource_loader, entity_type, **kwargs)
        self._es_host = kwargs.get(es_host, None)
        self._es_config = {"client": kwargs.get(es_client, None), "pid": os.getpid()}

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
        if clean:
            delete_index(
                self._app_namespace, self._es_index_name, self._es_host, self._es_client
            )
        entity_map = self._resource_loader.get_entity_map(self.type)

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


class EntityResolverUsingExactSearch(EntityResolverBase):

    def __init__(self, app_path, resource_loader, entity_type, er_config, **kwargs):
        super(EntityResolverUsingExactSearch, self).__init__(app_path, resource_loader, entity_type, **kwargs)
        self._normalizer = self._resource_loader.query_factory.normalize
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

        # list of canonical entities and their synonyms
        entities = entity_map.get("entities", [])

        """Fits a simple exact match entity resolution model when Elasticsearch is not available."""
        entity_map = self._resource_loader.get_entity_map(self.type)
        self._exact_match_mapping = self._process_entity_map(
            self.type, entity_map, self._normalizer
        )

    def _predict(self, entity):

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
