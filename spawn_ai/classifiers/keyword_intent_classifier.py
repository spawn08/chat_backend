from typing import Any, Optional, Text

from spawn_ai.components import Component
from spawn_ai.training_data import Message


class KeywordIntentClassifier(Component):
    name = "intent_classifier_keyword"

    provides = ["intent"]

    his = ["hello", "hi", "hey"]

    byes = ["bye", "goodbye"]

    def process(self, message: Message, **kwargs: Any) -> None:

        intent = {"name": self.parse(message.text), "confidence": 1.0}
        message.set("intent", intent,
                    add_to_output=True)

    def parse(self, text: Text) -> Optional[Text]:

        _text = text.lower()

        def is_present(x):
            return x in _text

        if any(map(is_present, self.his)):
            return "greet"
        elif any(map(is_present, self.byes)):
            return "goodbye"
        else:
            return None
