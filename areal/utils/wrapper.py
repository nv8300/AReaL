import inspect
from functools import partial
from typing import Callable, Dict, Optional


def wrapable(name: Optional[str] = None):
    def decorator(func):
        func.__wrap_meta__ = {
            "name": name or func.__name__,
        }
        return func

    return decorator


def wrap(target: object, source: object, transform: Optional[Callable] = None):
    for name, member in inspect.getmembers(source):
        if callable(member) and hasattr(member, "__wrap_meta__"):
            meta = member.__wrap_meta__
            method_name = meta["name"]

            if transform:
                setattr(
                    target,
                    method_name,
                    partial(
                        transform,
                        wrap_method_name=method_name,
                        wrap_original_method=member,
                    ),
                )
            else:
                setattr(target, method_name, member)


def wrap_get_method_name(kwargs) -> str:
    return kwargs["wrap_method_name"]


def wrap_get_method(kwargs) -> Callable:
    return kwargs["wrap_original_method"]


def wrap_remove_meta(kwargs) -> Dict:
    del kwargs["wrap_method_name"]
    del kwargs["wrap_original_method"]
    return kwargs
