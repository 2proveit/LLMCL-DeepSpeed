from .vanilla import *
from .ewc import *
from .gem import *
from .mtl import *
from .agem import *

TRAINERS = {
    "vanilla": VanillaTrainer,
    "ewc": EWCTrainer,
    "gem": GEMTrainer,
    'mtl': MultiTaskTrainer,
    'agem': AGEMTrainer
}
