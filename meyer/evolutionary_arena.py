import pickle
import random

from evoluionary_ais import EvoAi
from meyer import Game


def play_game(ais, n_rounds):
    new_game = Game(ais, n_rounds=n_rounds, terminal_output=False)
    new_game.start()
    new_game.get_winner().wins += 1


def battle(evo_ais, n_rounds):
    for i in range(len(evo_ais) // 2):
        play_game([evo_ais[2 * i], evo_ais[2 * i + 1]], n_rounds)


def evolve_generation(evo_ais: ["EvoAi"], n_rounds, battles=5, required_win_rate=.2):
    n_ais = len(evo_ais)

    # battle
    battle(evo_ais, n_rounds)
    for _ in range(battles - 1):
        random.shuffle(evo_ais)
        battle(evo_ais, n_rounds)

    # select
    survivors = list(filter(lambda a: a.wins > int(required_win_rate * battles), evo_ais))

    # mutate
    for ai in survivors:
        ai.mutate()
        ai.wins = 0

    # reproduce
    while len(survivors) < n_ais:
        child = random.choice(survivors).reproduce(random.choice(survivors))
        survivors.append(child)

    return survivors


def evolve(start_ais, n_rounds=80, battles=5, generations=1, required_win_rate=.2):
    survivors = start_ais.copy()
    for _ in range(generations):
        survivors = evolve_generation(survivors, n_rounds, battles, required_win_rate)
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
            play_game([evo1[i], evo2[i]], n_rounds)
    wins1 = sum(ai.wins for ai in evo1)
    wins2 = sum(ai.wins for ai in evo2)
    return wins1/(wins1 + wins2)


filename = "evo_data.pickle"


def load_evo_ais():
    with open(filename, "rb") as file:
        data_list = pickle.load(file)
    return [EvoAi.init_from_data(*data) for data in data_list]


def save_evo_ais(ai_list):
    with open(filename, "wb") as file:
        pickle.dump([ai.export_data() for ai in ai_list], file)


if __name__ == '__main__':
    LOAD = True
    if LOAD:
        initial_ais = load_evo_ais()
    else:
        initial_ais = [EvoAi() for _ in range(500)]
    survivor_list = initial_ais.copy()

    survivor_list = evolve(survivor_list, n_rounds=80, battles=10, generations=8, required_win_rate=.22)

    win_rate = head_to_head(survivor_list, initial_ais, n_rounds=60)
    print(win_rate)
    save_evo_ais(survivor_list)
