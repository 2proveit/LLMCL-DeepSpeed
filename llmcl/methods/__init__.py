from .vanilla import *
from .ewc import *
from .gem import *
from .mtl import *
from .agem import *
from .l2p import *
from .pp import *
from .ilora import *

TRAINERS = {
    "vanilla": VanillaTrainer,
    "ewc": EWCTrainer,
    "gem": GEMTrainer,
    'mtl': MultiTaskTrainer,
    'agem': AGEMTrainer,
    'l2p': L2PTrainer,
    'pp': PPTrainer,
    'ilora': ILoraTrainer,
}
