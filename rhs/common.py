from collections import OrderedDict

from numpyro.primitives import Message


type TraceType = OrderedDict[str, Message]

