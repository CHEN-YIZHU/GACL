from .er_baseline import ER
from .ewcpp import EWCpp
from .lwf import LwF
from .rainbow_memory import RM
from .mvp import MVP
from .GACL import GACL
from .SLDA import SLDA

__all__ = [
    "ER",
    "EWCpp",
    "LwF",
    "RM",
    "GACL",
    "MVP",
    "SLDA"
]

def get_method(name):
    name = name.lower()
    try:
        return {
            "er": ER,
            "ewcpp": EWCpp,
            "lwf": LwF,
            "rm": RM,
            "mvp": MVP,
            "gacl": GACL,
            "SLDA":SLDA
        }[name]
    except KeyError:
        raise NotImplementedError(f"Method {name} not implemented")