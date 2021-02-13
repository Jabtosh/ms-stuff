import numpy as np
import numpy.random as ran

valid_throws = [31, 32, 41, 42, 43, 51, 52, 53, 54, 61, 62, 63, 64, 65, 101, 102, 103, 104, 105, 106]

""""
15 mit zwei verschiedenen Zahlen. Zwei bestimmte verschiedene Zahlen haben die Wahrscheinlichkeit 2/6 * 1/6 = 2/36.
Ein Pasch hat die Wahrscheinlichkeit 1/6 * 1/6 = 1/36.
Die Summe der Wahrscheinlichkeiten ergibt: 15 * 2/36 + 6 * 1/36 = 30/36 + 6/36 = 1.
"""


class Game:
    def __init__(self, ais, tout=False):
        self.num_of_players = len(ais)
        self.throw_deciders = [ais[i][0] for i in range(len(ais))]
        self.claim_deciders = [ais[i][1] for i in range(len(ais))]
        self.names = [ais[i][2] for i in range(len(ais))]
        self.allow_terminal_output = tout
        self.round = 0
        self.points = np.zeros(self.num_of_players)
        self.meyer_count = np.zeros(self.num_of_players)
        self.current_player = 0
        self.last_throw = 0
        self.claims = []

    def player_turn(self):
        if self.throw_deciders[self.current_player](self.claims):
            self.last_throw = self.throw_dice()
            if self.check_meyer():
                self.points += 1
                self.points[self.current_player] -= 1
                self.meyer_count[self.current_player] += 1
                self.prep_new_round()
            else:
                self.claims.append(self.claim_deciders[self.current_player](self.claims, self.last_throw))
                self.print_out(f'roll {len(self.claims)}: player {self.current_player} '
                               f'rolled {self.last_throw} and claimed {self.claims[-1]}')
                self.next_player()
        else:
            self.doubt()

    def doubt(self):
        if self.last_throw == self.claims[-1] and (len(self.claims) == 1 or self.claims[-1] > self.claims[-2]):
            self.points[self.current_player] += 1
            self.print_out(f'player {self.current_player} doubted  {self.claims[-1]} and was wrong')
        else:
            self.points[self.current_player - 1] += 1
            self.print_out(f'player {self.current_player} doubted  {self.claims[-1]} and was right')
        self.prep_new_round()

    @staticmethod
    def throw_dice():
        d1 = ran.randint(1, 7)
        d2 = ran.randint(1, 7)
        g = max([d1, d2])
        s = min([d1, d2])
        if g == 2 and s == 1:
            return 200
        if g == s:
            return 100 + g
        else:
            return 10 * g + s

    def check_meyer(self):
        if self.last_throw == 200:  # MEYER
            return True
        else:
            return False

    def prep_new_round(self):
        self.last_throw = 0
        self.claims = []
        self.round += 1
        self.next_player()

    def next_player(self):
        self.current_player = (self.current_player + 1) % self.num_of_players  # go to next player

    def print_out(self, text):
        if self.allow_terminal_output:
            print(text)

    def print_points(self):
        print(self.points)
        print(self.meyer_count)

    def return_points(self):
        return self.points

    def return_winner(self):
        return self.names[np.argmin(self.points)]


def rank_from_throw(throw):
    return valid_throws.index(throw)


def throw_from_rank(rank):
    return valid_throws[rank]


def prob_from_throw(throw):
    if throw in valid_throws:
        if throw < 100:
            return 2 / 36
        else:
            return 1 / 36
    else:
        return 0.


def prob_from_rank(rank):
    if rank in range(20):
        if rank < 14:
            return 2 / 36
        else:
            return 1 / 36
    else:
        return 0.


def cum_prob_from_rank(rank):
    """including the given rank"""
    if rank in range(20):
        if rank < 14:
            return (2 / 36) * (1 + rank)
        else:
            return (2 / 36) * 14 + (1 / 36) * (1 + rank - 14)
    else:
        return 0.


def cum_prob_from_throw(throw):
    return cum_prob_from_rank(rank_from_throw(throw))


def dummy_throw(claims):
    if not claims:
        return True
    if (claims[-1] >= 101) or (claims[-1] not in valid_throws):
        return False
    else:
        return True


def dummy_claims(claims, throw):
    if not claims:
        return throw
    if throw > claims[-1]:
        return throw
    elif claims[-1] == valid_throws[-1]:
        return claims[-1]
    else:
        return throw_from_rank(rank_from_throw(claims[-1]) + 1)


def test_ai_throw(claims):
    if not claims:
        return True
    elif len(claims) == 1:
        return True
    elif (claims[-1] >= 102) or (claims[-1] not in valid_throws) or (
            rank_from_throw(claims[-1]) == rank_from_throw(claims[-2]) + 1):
        return False
    else:
        return True


def test_ai_claims(claims, throw):
    if not claims:
        return throw
    if throw > claims[-1]:
        return throw
    elif claims[-1] == valid_throws[-1]:
        return claims[-1]
    else:
        return throw_from_rank(rank_from_throw(claims[-1]) + 1)


####################################### Thowa AI #########################################

class ThowaAI():
    def __init__(self):
        return

    def throw_decider(self, claims):
        if not claims:
            return True
        if claims[-1] not in valid_throws or claims[-1] == valid_throws[-1]:
            return False
        else:
            return True

    def claim_decider(self, claims, throw):
        if not claims:
            return throw
        if throw > claims[-1]:  # only bluff, if necessary
            return throw
        else:
            return bluff(claims)


# Ermittelt eine größere Zahl als der letzte claim
def bluff(claims):
    higher_throws = valid_throws[- (len(valid_throws) - 1 - valid_throws.index(claims[-1])):]
    if len(higher_throws) == 0:  # wenn es keine Höheren gibt, irgendwas sagen
        return valid_throws[-1]
    elif len(higher_throws) == 1:  #
        return higher_throws[0]
    return valid_throws[ran.randint(0, len(higher_throws) - 1)]


##################################### Main Code ######################################

def playGame(ais, turns):
    new_game = Game(ais, tout=False)
    for turn in range(turns):
        new_game.player_turn()
    points = new_game.return_points()
    print(str(points / sum(points) * len(ais)))
    print(' Winner: ' + new_game.return_winner())


dummyAI = (dummy_throw, dummy_claims, 'Dummy')
testAI = (test_ai_throw, test_ai_claims, 'Test')

new_thowaAI = ThowaAI()
thowaAI = [new_thowaAI.throw_decider, new_thowaAI.claim_decider, 'Thowa']

playGame([dummyAI, testAI], 100000)
playGame([dummyAI, thowaAI], 100000)
playGame([thowaAI, testAI], 100000)
