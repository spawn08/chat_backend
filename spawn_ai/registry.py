"""This is a somewhat delicate package. It contains all registered components
and preconfigured templates.

Hence, it imports all of the components. To avoid cycles, no component should
import this in module scope."""

import typing
from typing import Any, Dict, List, Optional, Text, Type

from spawn_ai import utils
from spawn_ai.classifiers.embedding_intent_classifier import \
    EmbeddingIntentClassifier
from spawn_ai.classifiers.keyword_intent_classifier import \
    KeywordIntentClassifier
from spawn_ai.classifiers.mitie_intent_classifier import MitieIntentClassifier
from spawn_ai.classifiers.sklearn_intent_classifier import \
    SklearnIntentClassifier
from spawn_ai.extractors.crf_entity_extractor import CRFEntityExtractor
from spawn_ai.extractors.duckling_http_extractor import DucklingHTTPExtractor
from spawn_ai.extractors.entity_synonyms import EntitySynonymMapper
from spawn_ai.extractors.mitie_entity_extractor import MitieEntityExtractor
from spawn_ai.extractors.spacy_entity_extractor import SpacyEntityExtractor
from spawn_ai.featurizers.count_vectors_featurizer import \
    CountVectorsFeaturizer
from spawn_ai.featurizers.mitie_featurizer import MitieFeaturizer
from spawn_ai.featurizers.ngram_featurizer import NGramFeaturizer
from spawn_ai.featurizers.regex_featurizer import RegexFeaturizer
from spawn_ai.featurizers.spacy_featurizer import SpacyFeaturizer
from spawn_ai.model import Metadata
from spawn_ai.tokenizers.jieba_tokenizer import JiebaTokenizer
from spawn_ai.tokenizers.mitie_tokenizer import MitieTokenizer
from spawn_ai.tokenizers.spacy_tokenizer import SpacyTokenizer
from spawn_ai.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from spawn_ai.utils.mitie_utils import MitieNLP
from spawn_ai.utils.spacy_utils import SpacyNLP

if typing.TYPE_CHECKING:
    from spawn_ai.components import Component
    from spawn_ai.config import RasaNLUModelConfig, RasaNLUModelConfig

# Classes of all known components. If a new component should be added,
# its class name should be listed here.
component_classes = [
    SpacyNLP, MitieNLP,
    SpacyEntityExtractor, MitieEntityExtractor,
    CRFEntityExtractor, DucklingHTTPExtractor,
    EntitySynonymMapper,
    SpacyFeaturizer, MitieFeaturizer, NGramFeaturizer, RegexFeaturizer,
    CountVectorsFeaturizer,
    MitieTokenizer, SpacyTokenizer, WhitespaceTokenizer, JiebaTokenizer,
    SklearnIntentClassifier, MitieIntentClassifier, KeywordIntentClassifier,
    EmbeddingIntentClassifier
]

# Mapping from a components name to its class to allow name based lookup.
registered_components = {c.name: c for c in component_classes}

# To simplify usage, there are a couple of model templates, that already add
# necessary components in the right order. They also implement
# the preexisting `backends`.
registered_pipeline_templates = {
    "spacy_sklearn": [
        "nlp_spacy",
        "tokenizer_spacy",
        "intent_featurizer_spacy",
        "intent_entity_featurizer_regex",
        "ner_crf",
        "ner_synonyms",
        "intent_classifier_sklearn",
    ],
    "keyword": [
        "intent_classifier_keyword",
    ],
    "tensorflow_embedding": [
        "tokenizer_whitespace",
        "ner_crf",
        "ner_synonyms",
        "intent_featurizer_count_vectors",
        "intent_classifier_tensorflow_embedding"
    ]
}


def pipeline_template(s: Text) -> Optional[List[Dict[Text, Text]]]:
    components = registered_pipeline_templates.get(s)

    if components:
        # converts the list of components in the configuration
        # format expected (one json object per component)
        return [{"name": c} for c in components]

    else:
        return None


def get_component_class(component_name: Text) -> Type['Component']:
    """Resolve component name to a registered components class."""

    if component_name not in registered_components:
        try:
            return utils.class_from_module_path(component_name)
        except Exception:
            raise Exception(
                "Failed to find component class for '{}'. Unknown "
                "component name. Check your configured pipeline and make "
                "sure the mentioned component is not misspelled. If you "
                "are creating your own component, make sure it is either "
                "listed as part of the `component_classes` in "
                "`spawn_ai.registry.py` or is a proper name of a class "
                "in a module.".format(component_name))
    return registered_components[component_name]


def load_component_by_name(component_name: Text,
                           model_dir: Text,
                           metadata: Metadata,
                           cached_component: Optional['Component'],
                           **kwargs: Any
                           ) -> Optional['Component']:
    """Resolves a component and calls its load method to init it based on a
    previously persisted model."""

    component_clz = get_component_class(component_name)
    return component_clz.load(model_dir, metadata, cached_component, **kwargs)


def create_component_by_name(component_name: Text,
                             config: 'RasaNLUModelConfig'
                             ) -> Optional['Component']:
    """Resolves a component and calls it's create method to init it based on a
    previously persisted model."""

    component_clz = get_component_class(component_name)
    return component_clz.create(config)
