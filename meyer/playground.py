from basic_ais import MinimalAi, CounterMinimalAi, ThowaAi, SimpleAi
from evolutionary_arena import load_evo_ais
from meyer import Game
from meyer_optimizer import OptAI


def play_game(ais, n_rounds):
    print([ai.get_name() for ai in ais])
    new_game = Game(ais, n_rounds=n_rounds, terminal_output=False)
    new_game.start()
    points = new_game.points
    print(points / sum(points))
    print(f'Winner: {new_game.get_winner()}')
    print(f'----------------')


dummy_ai = MinimalAi('Dummy')
counter_ai = CounterMinimalAi('Test')
thowa_ai = ThowaAi('Thowa')
opt_ai = OptAI('Opto')
simple_ai = SimpleAi('Basic')
evo_ais = load_evo_ais()

N = 50000

play_game([evo_ais[0], evo_ais[1]], N)
play_game([opt_ai, evo_ais[2]], N)
play_game([counter_ai, simple_ai], N)
play_game([opt_ai, simple_ai], N)
play_game([dummy_ai, simple_ai], N)
play_game([thowa_ai, simple_ai], N)
