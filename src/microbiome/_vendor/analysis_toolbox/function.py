# Vendored from analysis_toolbox (function.py) at commit 7f96772069025d136f802c364879703b086dec94 (plus uncommitted working-tree changes).
# Only definitions reached by this project are included; see _vendor/README.md.


import inspect


def is_mod_function(mod, func):
    """
    checks that func is a function defined in module mod
    """
    return inspect.isfunction(func) and inspect.getmodule(func) == mod


def list_functions(mod, mod_function=True):
    """
    list of functions defined in module mod

    Usage
    ======
    print(list_functions(something))
    """
    if mod_function:
        return [func.__name__ for func in mod.__dict__.values()
                if is_mod_function(mod, func)]

    else:
        return [func.__name__ for func in mod.__dict__.values()
                if inspect.isfunction(func)]
