from LogicGates.Evaluation import Grouper
from LogicGates.Gates import Bit, Inv, And, Or, Xor
from LogicGates.Solver import Solver, HSolver


def test_solver():
    # Id
    expected = "0: 0; 1: 1"
    solution = Solver(expected).solve()
    assert solution.cost == 0

    # Inv
    expected = "0: 1; 1: 0"
    solution = Solver(expected).solve()
    assert solution.cost == 1

    # Nand
    expected = "00: 1; 01: 1; 10: 1; 11: 0"
    solution = Solver(expected).solve()
    assert solution.cost == 1

    # And
    expected = "00: 0; 01: 0; 10: 0; 11: 1"
    solution = Solver(expected).solve()
    assert solution.cost == 2

    # Or
    expected = "00: 0; 01: 1; 10: 1; 11: 1"
    solution = Solver(expected).solve()
    assert solution.cost == 3

    # Xor
    expected = "00: 0; 01: 1; 10: 1; 11: 0"
    solution = Solver(expected).solve()
    assert solution.cost == 4

    # HalfAdd
    expected = "00: 00; 01: 01; 10: 01; 11: 10"
    solution = Solver(expected).solve()
    assert solution.cost == 5


def test_hsolver():
    # Id
    expected = "0: 0; 1: 1"
    solution = HSolver(expected).solve()
    assert solution.cost == 0

    # Inv
    expected = "0: 1; 1: 0"
    solution = HSolver(expected).solve()
    assert solution.cost == 1

    # Nand
    expected = "00: 1; 01: 1; 10: 1; 11: 0"
    solution = HSolver(expected).solve()
    assert solution.cost == 1

    # And
    expected = "00: 0; 01: 0; 10: 0; 11: 1"
    solution = HSolver(expected).solve()
    assert solution.cost == 2

    # Or
    expected = "00: 0; 01: 1; 10: 1; 11: 1"
    solution = HSolver(expected).solve()
    assert solution.cost == 3

    # Xor
    expected = "00: 0; 01: 1; 10: 1; 11: 0"
    solution = HSolver(expected).solve()
    assert solution.cost == 4

    # HalfAdd
    expected = "00: 00; 01: 01; 10: 01; 11: 10"
    solution = HSolver(expected).solve()
    assert solution.cost == 5


def test_grouper():
    # Inv
    roots = Bit.init_n(1)
    expected = "0: 1; 1: 0"
    solution = Solver(expected).solve()
    grouper = Grouper(roots, solution.output)
    grouper.simplify()
    assert grouper.costs() == solution.cost
    assert len(grouper.outputs) == 1
    assert isinstance(grouper.outputs[0], Inv)

    # And
    roots = Bit.init_n(2)
    expected = "00: 0; 01: 0; 10: 0; 11: 1"
    solution = Solver(expected).solve()
    grouper = Grouper(roots, solution.output)
    grouper.simplify()
    assert grouper.costs() == solution.cost
    assert len(grouper.outputs) == 1
    assert isinstance(grouper.outputs[0], And)

    # Or
    roots = Bit.init_n(2)
    expected = "00: 0; 01: 1; 10: 1; 11: 1"
    solution = Solver(expected).solve()
    grouper = Grouper(roots, solution.output)
    grouper.simplify()
    assert grouper.costs() == solution.cost
    assert len(grouper.outputs) == 1
    assert isinstance(grouper.outputs[0], Or)

    # Xor
    roots = Bit.init_n(2)
    expected = "00: 0; 01: 1; 10: 1; 11: 0"
    solution = Solver(expected).solve()
    grouper = Grouper(roots, solution.output)
    grouper.simplify()
    assert grouper.costs() == solution.cost
    assert len(grouper.outputs) == 1
    assert isinstance(grouper.outputs[0], Xor)

    # HalfAdd
    roots = Bit.init_n(2)
    expected = "00: 00; 01: 01; 10: 01; 11: 10"
    solution = Solver(expected).solve()
    grouper = Grouper(roots, solution.output)
    grouper.simplify()
    assert grouper.costs() == solution.cost
    assert len(grouper.outputs) == 2
