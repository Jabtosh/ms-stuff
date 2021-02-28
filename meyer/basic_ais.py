""" ---------------------------------------- Basic AIs --------------------------------------------------------- """
import numpy.random as ran

from constants import V
from meyer import AiBase


class DummyAI(AiBase):
    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        return self.name

    def doubt_decider(self, claims: tuple, n_players: int, n_rounds_remaining: int) -> bool:
        if not claims:
            return False
        if (claims[-1] >= 15) or (claims[-1] not in V):
            return True
        else:
            return False

    def claim_decider(self, claims: tuple, last_throw: int) -> int:
        if not claims:
            return last_throw
        if last_throw > claims[-1]:
            return last_throw
        elif claims[-1] == V[-1]:
            return claims[-1]
        else:
            return claims[-1] + 1


class TestAI(AiBase):
    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        return self.name

    def doubt_decider(self, claims: tuple, n_players: int, n_rounds_remaining: int) -> bool:
        if not claims:
            return False
        elif len(claims) == 1:
            return False
        elif (claims[-1] >= 16) or (claims[-1] not in V) or (claims[-1] == claims[-2] + 1):
            return True
        else:
            return False

    def claim_decider(self, claims: tuple, last_throw: int) -> int:
        if not claims:
            return last_throw
        if last_throw > claims[-1]:
            return last_throw
        elif claims[-1] == V[-1]:
            return claims[-1]
        else:
            return claims[-1] + 1


""" -------------------------------------------- Thowa AI ------------------------------------------------------ """


class ThowaAI(AiBase):
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    @classmethod
    def doubt_decider(cls, claims, n_players, n_rounds_remaining):
        if not claims:
            return False
        elif claims[-1] not in V or claims[-1] == V[-1]:
            return True
        else:
            return False

    @classmethod
    def claim_decider(cls, claims, last_throw):
        if not claims:
            return last_throw
        if last_throw > claims[-1]:  # only bluff, if necessary
            return last_throw
        else:
            return cls.basic_bluff(claims)

    @staticmethod
    def basic_bluff(claims):
        higher_throws = V[claims[-1] + 1:]
        if len(higher_throws) == 0:  # claim anything
            return V[-1]
        elif len(higher_throws) == 1:
            return higher_throws[0]
        return V[ran.randint(higher_throws[0], higher_throws[-1] + 1)]
