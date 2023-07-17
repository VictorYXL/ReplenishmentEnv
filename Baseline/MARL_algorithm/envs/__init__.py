from functools import partial
from .multiagentenv import MultiAgentEnv
from .replenishment import ReplenishmentEnv
from .simple_multi_echelon import SimpleMultiEchelonEnv
import sys
import os


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}

REGISTRY["replenishment"] = partial(env_fn, env=ReplenishmentEnv)
REGISTRY["simple_multi_echelon"] = partial(env_fn, env=SimpleMultiEchelonEnv)

