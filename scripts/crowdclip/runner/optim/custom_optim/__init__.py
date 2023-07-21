from .lookahead import Lookahead
from .radam import RAdam
from .ranger import Ranger
from .ranger21 import Ranger21
from .adai import Adai
from .adaiw import AdaiW

AVAI_OPTIMS = [
    "adam",
    "amsgrad",
    "sgd",
    "rmsprop",
    "radam",
    "adamw",
    "ranger",
    "ranger21",
    "adai",
    "adaiw"
]