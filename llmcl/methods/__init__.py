from .vanilla import *
from .ewc import *
# from .gem import *

TRAINERS = {
    "vanilla": VanillaTrainer,
    "ewc": EWCTrainer,
    # "gem": GEMTrainer,
}
