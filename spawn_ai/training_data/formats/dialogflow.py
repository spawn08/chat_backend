import logging
import os
import typing
from typing import Any, Text

from spawn_ai import utils
from spawn_ai.training_data.formats.readerwriter import TrainingDataReader
from spawn_ai.training_data.util import transform_entity_synonyms

if typing.TYPE_CHECKING:
    from spawn_ai.training_data import TrainingData

logger = logging.getLogger(__name__)

DIALOGFLOW_PACKAGE = "dialogflow_package"
DIALOGFLOW_AGENT = "dialogflow_agent"
DIALOGFLOW_INTENT = "dialogflow_intent"
DIALOGFLOW_INTENT_EXAMPLES = "dialogflow_intent_examples"
DIALOGFLOW_ENTITIES = "dialogflow_entities"
DIALOGFLOW_ENTITY_ENTRIES = "dialogflow_entity_entries"


class DialogflowReader(TrainingDataReader):
    def read(self, fn: Text, **kwargs: Any) -> 'TrainingData':
        """Loads training data stored in the Dialogflow data format."""
        from spawn_ai.training_data import TrainingData

        language = kwargs["language"]
        fformat = kwargs["fformat"]

        if fformat not in {DIALOGFLOW_INTENT, DIALOGFLOW_ENTITIES}:
            raise ValueError("fformat must be either {}, or {}"
                             "".format(DIALOGFLOW_INTENT, DIALOGFLOW_ENTITIES))

        root_js = utils.read_json_file(fn)
        examples_js = self._read_examples_js(fn, language, fformat)

        if not examples_js:
            logger.warning("No training examples found for dialogflow file {}!"
                           "".format(fn))
            return TrainingData()
        elif fformat == DIALOGFLOW_INTENT:
            return self._read_intent(root_js, examples_js)
        elif fformat == DIALOGFLOW_ENTITIES:
            return self._read_entities(examples_js)

    def _read_intent(self, intent_js, examples_js):
        """Reads the intent and examples from respective jsons."""
        from spawn_ai.training_data import Message, TrainingData

        intent = intent_js.get("name")

        training_examples = []
        for ex in examples_js:
            text, entities = self._join_text_chunks(ex['data'])
            training_examples.append(Message.build(text, intent, entities))

        return TrainingData(training_examples)

    def _join_text_chunks(self, chunks):
        """Combines text chunks and extracts entities."""

        utterance = ""
        entities = []
        for chunk in chunks:
            entity = self._extract_entity(chunk, len(utterance))
            if entity:
                entities.append(entity)
            utterance += chunk["text"]

        return utterance, entities

    @staticmethod
    def _extract_entity(chunk, current_offset):
        """Extract an entity from a chunk if present."""

        entity = None
        if "meta" in chunk or "alias" in chunk:
            start = current_offset
            text = chunk['text']
            end = start + len(text)
            entity_type = chunk.get("alias", chunk["meta"])
            if entity_type != u'@sys.ignore':
                entity = utils.build_entity(start, end, text, entity_type)

        return entity

    @staticmethod
    def _read_entities(examples_js):
        from spawn_ai.training_data import TrainingData

        entity_synonyms = transform_entity_synonyms(examples_js)
        return TrainingData([], entity_synonyms)

    @staticmethod
    def _read_examples_js(fn, language, fformat):
        """Infer and load the example file based on the root
        filename and root format."""

        if fformat == DIALOGFLOW_INTENT:
            examples_type = "usersays"
        else:
            examples_type = "entries"
        examples_fn_ending = "_{}_{}.json".format(examples_type, language)
        examples_fn = fn.replace(".json", examples_fn_ending)
        if os.path.isfile(examples_fn):
            return utils.read_json_file(examples_fn)
        else:
            return None

    def reads(self, s, **kwargs):
        raise NotImplementedError
