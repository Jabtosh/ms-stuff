import pickle
import random

from constants import V
from evolutionary_ais import EvoAi, DoubtArray
from meyer import Game
from meyer_optimizer import do_doubt, best_lie


def play_game(ais, n_rounds):
    new_game = Game(ais, n_rounds=n_rounds, terminal_output=False)
    new_game.start()
    new_game.get_winner().wins += 1


def battle(evo_ais, n_rounds):
    for i in range(len(evo_ais) // 2):
        play_game([evo_ais[2 * i], evo_ais[2 * i + 1]], n_rounds)


def evolve_generation(evo_ais: ["EvoAi"], n_rounds, battles=6, required_wins=2):
    n_ais = len(evo_ais)

    # battle
    for _ in range(battles):
        random.shuffle(evo_ais)
        battle(evo_ais, n_rounds)

    # select
    survivors = list(filter(lambda a: a.wins >= int(required_wins), evo_ais))
    print(f"{len(survivors)} / {n_ais} survived")

    # reproduce
    survivors.sort(key=lambda x: x.wins, reverse=True)
    survivors.append(EvoAi())
    i = 0
    while len(survivors) < n_ais:
        child = survivors[i].reproduce(survivors[i+1])
        survivors.append(child)
        # survivors[i].doubt_array.M_RATE *= .75
        # survivors[i].lie_array.M_RATE *= .75
        i += 2

    # mutate
    for ai in survivors:
        ai.wins = 0
        ai.mutate()

    return survivors


def evolve(start_ais, n_rounds=80, battles=6, generations=1, required_wins=2):
    survivors = start_ais
    for i in range(generations):
        print(f"Generation {i}")
        survivors = evolve_generation(survivors, n_rounds, battles, required_wins)
    return survivors


def head_to_head(evo1: [EvoAi], evo2: [EvoAi], n_players=2, n_rounds=80, battles=5):
    if not len(evo1) == len(evo2):
        raise IndexError("Lists must have the same length.")
    for i in range(len(evo1)):
        evo1[i].wins = 0
        evo2[i].wins = 0
    for _ in range(battles):
        random.shuffle(evo2)
        for i in range(len(evo1)):
            play_game([evo1[i], evo2[i]], n_rounds//2)
            play_game([evo2[i], evo1[i]], n_rounds//2)
    wins1 = sum(ai.wins for ai in evo1)
    wins2 = sum(ai.wins for ai in evo2)
    for i in range(len(evo1)):
        evo1[i].wins = 0
        evo2[i].wins = 0
    return wins1/(wins1 + wins2)


filename = "evo_data.pickle"


def load_evo_ais():
    with open(filename, "rb") as file:
        data_list = pickle.load(file)
    return [EvoAi.init_from_data(*data) for data in data_list]


def save_evo_ais(ai_list):
    with open(filename, "wb") as file:
        pickle.dump([ai.export_data() for ai in ai_list], file)


def transform_ai_to_opt(ai):
    for n_rounds in range(DoubtArray.R):
        for claim_m2 in V:
            for claim_m1 in V[claim_m2 + 1:]:
                ai.doubt_array[n_rounds, 0, claim_m1 - 1, claim_m2] = do_doubt(2, claim_m2, claim_m1, n_rounds)
                ai.lie_array[n_rounds, 0, claim_m1 - 1, claim_m2] = \
                    1. if best_lie(2, claim_m2, n_rounds) == claim_m1 else 0.


if __name__ == '__main__':
    LOAD = True
    if LOAD:
        initial_ais = load_evo_ais()
    else:
        initial_ais = [EvoAi() for _ in range(1000)]
    survivor_list = [ai.copy() for ai in initial_ais]

    survivor_list = evolve(survivor_list, generations=8, n_rounds=42, battles=12, required_wins=4)

    win_rate = head_to_head(survivor_list, initial_ais, n_rounds=50, battles=8)
    print(win_rate)
    save_evo_ais(survivor_list)
