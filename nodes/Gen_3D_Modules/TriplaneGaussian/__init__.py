import importlib
from .utils.typing import *
from ..._vendor_paths import import_module as _vendor_import

def find(cls_string) -> Type:
    cls_full_name = cls_string.split(".")
    module_string = ".".join(cls_full_name[:-1])
    cls_name = cls_full_name[-1]
    print(module_string)
    module = _vendor_import(module_string)
    cls = getattr(module, cls_name)
    return cls