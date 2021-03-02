from abc import ABCMeta, abstractmethod

import numpy as np
import numpy.random as ran

from constants import RNG_MAP, V


class AiBase(metaclass=ABCMeta):
    def __init__(self, name: str):
        self.name = name
        self.wins = 0

    def get_name(self) -> str:
        return self.name

    @abstractmethod
    def doubt_decider(self, claims: tuple, n_players: int, n_rounds_remaining: int) -> bool:
        ...

    @abstractmethod
    def claim_decider(self, claims: tuple, last_throw: int, n_players: int, n_rounds_remaining: int) -> int:
        ...

    def __repr__(self):
        return self.get_name()


class Game:
    def __init__(self, ais: [AiBase], n_rounds: int, terminal_output: bool = False):
        self.n_rounds_remaining = n_rounds - 1
        self.terminal_output = terminal_output
        self.n_players = len(ais)
        self.doubt_deciders = tuple(ais[i].doubt_decider for i in range(self.n_players))
        self.claim_deciders = tuple(ais[i].claim_decider for i in range(self.n_players))
        self.ais = ais
        self.points = np.zeros(self.n_players)
        self.meyer_count = np.zeros(self.n_players)
        self.throw_generator = self.generate_throws(self.n_players * self.n_rounds_remaining)
        self.player_index = 0
        self.last_throw = 0
        # Latest claim has index -1
        self.claims = (0, )

    def start(self):
        while self.n_rounds_remaining >= 0:
            self.player_turn()

    def player_turn(self):
        doubter = self.doubt_deciders[self.player_index]
        if self.move_invalid() or (self.claims[1:] and doubter(self.claims, self.n_players, self.n_rounds_remaining)):
            self.doubt()
        else:
            self.throw()

    def throw(self):
        self.last_throw = self.throw_generator.__next__()
        if self.is_meyer():
            self.points += 1
            self.points[self.player_index] -= 1
            self.meyer_count[self.player_index] += 1
            self.prepare_new_round()
        else:
            new_claim = self.claim_deciders[self.player_index](self.claims, self.last_throw, self.n_players,
                                                               self.n_rounds_remaining)
            self.claims = (*self.claims, new_claim)
            self.print_out(f'roll {len(self.claims)}: player {self.player_index} '
                           f'rolled {self.last_throw} and claimed {self.claims[-1]}')
            self.next_player()

    def doubt(self):
        if self.last_throw == self.claims[-1] and self.last_throw > self.claims[-2]:
            # false accusation
            self.points[self.player_index] += 1
        else:
            # successful doubt
            self.points[self.player_index - 1] += 1
        self.prepare_new_round()

    def move_invalid(self) -> bool:
        return (self.claims[-1] not in V) or (len(self.claims) > 1 and self.claims[-2] >= self.claims[-1])

    def prepare_new_round(self):
        self.last_throw = 0
        self.claims = (0, )
        self.n_rounds_remaining -= 1
        self.next_player()

    def next_player(self):
        """ Increment player index. """
        self.player_index = (self.player_index + 1) % self.n_players

    @staticmethod
    def generate_throws(size=10000):
        """ Convenience generator for random numbers.
        :param size: estimate for the number of needed random numbers
        :returns: RNG_MAP[ran.randint(36)]
        """
        while True:
            throw_indices = ran.randint(36, size=size)
            for throw_index in throw_indices:
                yield RNG_MAP[throw_index]

    def is_meyer(self) -> bool:
        if self.last_throw == 21:
            return True
        else:
            return False

    def print_out(self, text):
        if self.terminal_output:
            print(text)

    def get_winner(self) -> AiBase:
        return self.ais[self.points.argmin()]
