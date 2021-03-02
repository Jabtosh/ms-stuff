from basic_ais import MinimalAi, CounterMinimalAi, ThowaAi, SimpleAi
from evoluionary_ais import EvoAi
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
evo1 = EvoAi()
evo2 = EvoAi()

N = 50000

play_game([evo1, evo2], N)
play_game([counter_ai, simple_ai], N)
play_game([opt_ai, simple_ai], N)
play_game([dummy_ai, simple_ai], N)
play_game([thowa_ai, simple_ai], N)
