import collections.abc

"""
    Deep update dict without remove origin items
"""
def deep_update(dic1: dict, dic2: dict) -> dict:
    for key, value in dic2.items():
        if isinstance(value, collections.abc.Mapping):
            dic1[key] = deep_update(dic1.get(key, {}), value)
        else:
            dic1[key] = value
    return dic1