import os
import typing
from typing import Any, Dict, List, Optional, Text

from spawn_ai.components import Component
from spawn_ai.config import RasaNLUModelConfig
from spawn_ai.model import Metadata
from spawn_ai.training_data import Message, TrainingData

if typing.TYPE_CHECKING:
    import mitie

MITIE_MODEL_FILE_NAME = "intent_classifier.dat"


class MitieIntentClassifier(Component):
    name = "intent_classifier_mitie"

    provides = ["intent"]

    requires = ["tokens", "mitie_feature_extractor", "mitie_file"]

    def __init__(self,
                 component_config: Dict[Text, Any] = None,
                 clf=None
                 ) -> None:
        """Construct a new intent classifier using the MITIE framework."""

        super(MitieIntentClassifier, self).__init__(component_config)

        self.clf = clf

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["mitie"]

    def train(self, training_data: TrainingData, cfg: RasaNLUModelConfig,
              **kwargs: Any) -> None:
        import mitie

        model_file = kwargs.get("mitie_file")
        if not model_file:
            raise Exception("Can not run MITIE entity extractor without a "
                            "language model. Make sure this component is "
                            "preceeded by the 'nlp_mitie' component.")

        trainer = mitie.text_categorizer_trainer(model_file)
        trainer.num_threads = kwargs.get("num_threads", 1)

        for example in training_data.intent_examples:
            tokens = self._tokens_of_message(example)
            trainer.add_labeled_text(tokens, example.get("intent"))

        if training_data.intent_examples:
            # we can not call train if there are no examples!
            self.clf = trainer.train()

    def process(self, message: Message, **kwargs: Any) -> None:

        mitie_feature_extractor = kwargs.get("mitie_feature_extractor")
        if not mitie_feature_extractor:
            raise Exception("Failed to train 'intent_featurizer_mitie'. "
                            "Missing a proper MITIE feature extractor.")

        if self.clf:
            token_strs = self._tokens_of_message(message)
            intent, confidence = self.clf(token_strs, mitie_feature_extractor)
        else:
            # either the model didn't get trained or it wasn't
            # provided with any data
            intent = None
            confidence = 0.0

        message.set("intent", {"name": intent, "confidence": confidence},
                    add_to_output=True)

    @staticmethod
    def _tokens_of_message(message):
        return [token.text for token in message.get("tokens", [])]

    @classmethod
    def load(cls,
             model_dir: Optional[Text] = None,
             model_metadata: Optional[Metadata] = None,
             cached_component: Optional['MitieIntentClassifier'] = None,
             **kwargs: Any
             ) -> 'MitieIntentClassifier':
        import mitie

        meta = model_metadata.for_component(cls.name)
        file_name = meta.get("classifier_file", MITIE_MODEL_FILE_NAME)

        if not file_name:
            return cls(meta)
        classifier_file = os.path.join(model_dir, file_name)
        if os.path.exists(classifier_file):
            classifier = mitie.text_categorizer(classifier_file)
            return cls(meta, classifier)
        else:
            return cls(meta)

    def persist(self, model_dir: Text) -> Dict[Text, Any]:
        import os

        if self.clf:
            classifier_file = os.path.join(model_dir, MITIE_MODEL_FILE_NAME)
            self.clf.save_to_disk(classifier_file, pure_model=True)
            return {"classifier_file": MITIE_MODEL_FILE_NAME}
        else:
            return {"classifier_file": None}