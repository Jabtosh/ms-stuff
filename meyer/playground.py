from basic_ais import DummyAI, TestAI, ThowaAI
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


dummyAI = DummyAI('Dummy')
testAI = TestAI('Test')
thowaAI = ThowaAI('Thowa')
optAI = OptAI('Opto')

play_game([optAI, testAI], 100000)
play_game([dummyAI, testAI], 100000)
play_game([thowaAI, testAI], 100000)
