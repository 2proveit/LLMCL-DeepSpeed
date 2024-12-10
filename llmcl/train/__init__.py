from sympy import im
from get_args import *
from .methods.vanilla import *
from .methods.ewc import *

TRAINERS = {
    "vanilla": VanillaTrainer,
    "ewc": EWCTrainer
}

