from datetime import datetime


def cached_function(func):

    def inner(*args):
        if args not in inner.cache.keys():
            inner.cache[args] = func(*args)
        return inner.cache[args]

    inner.cache = {}
    return inner


def cached_function_alternative(func):
    func.cache = {}

    def inner(*args):
        if args not in func.cache.keys():
            func.cache[args] = func(*args)
        return func.cache[args]

    return inner
