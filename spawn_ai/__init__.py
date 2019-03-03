import logging

import spawn_ai.version

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = spawn_ai.version.__version__
