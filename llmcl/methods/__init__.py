from .vanilla import *
from .ewc import *
from .gem import *
from .mtl import *

TRAINERS = {
    "vanilla": VanillaTrainer,
    "ewc": EWCTrainer,
    "gem": GEMTrainer,
    'mtl': MultiTaskTrainer
}
