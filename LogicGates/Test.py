from LogicGates.Evaluation import Grouper
from LogicGates.Gates import Bit, Inv, And, Or, Xor
from LogicGates.Solver import Solver


def test_solver():
    # Inv
    roots = [Bit()]
    expected = "0: 1; 1: 0"
    solution = Solver(roots, expected).solve()
    assert solution.cost == 1

    # And
    roots = [Bit(), Bit()]
    expected = "00: 0; 01: 0; 10: 0; 11: 1"
    solution = Solver(roots, expected).solve()
    assert solution.cost == 2

    # Or
    roots = [Bit(), Bit()]
    expected = "00: 0; 01: 1; 10: 1; 11: 1"
    solution = Solver(roots, expected).solve()
    assert solution.cost == 3

    # Xor
    roots = [Bit(), Bit()]
    expected = "00: 0; 01: 1; 10: 1; 11: 0"
    solution = Solver(roots, expected).solve()
    assert solution.cost == 4

    # HalfAdd
    roots = [Bit(), Bit()]
    expected = "00: 00; 01: 01; 10: 01; 11: 10"
    solution = Solver(roots, expected).solve()
    assert solution.cost == 5


def test_grouper():
    # Inv
    roots = [Bit()]
    expected = "0: 1; 1: 0"
    solution = Solver(roots, expected).solve()
    grouper = Grouper(roots, solution.output)
    grouper.simplify()
    assert grouper.costs() == solution.cost
    assert len(grouper.outputs) == 1
    assert isinstance(grouper.outputs[0], Inv)

    # And
    roots = [Bit(), Bit()]
    expected = "00: 0; 01: 0; 10: 0; 11: 1"
    solution = Solver(roots, expected).solve()
    grouper = Grouper(roots, solution.output)
    grouper.simplify()
    assert grouper.costs() == solution.cost
    assert len(grouper.outputs) == 1
    assert isinstance(grouper.outputs[0], And)

    # Or
    roots = [Bit(), Bit()]
    expected = "00: 0; 01: 1; 10: 1; 11: 1"
    solution = Solver(roots, expected).solve()
    grouper = Grouper(roots, solution.output)
    grouper.simplify()
    assert grouper.costs() == solution.cost
    assert len(grouper.outputs) == 1
    assert isinstance(grouper.outputs[0], Or)

    # Xor
    roots = [Bit(), Bit()]
    expected = "00: 0; 01: 1; 10: 1; 11: 0"
    solution = Solver(roots, expected).solve()
    grouper = Grouper(roots, solution.output)
    grouper.simplify()
    assert grouper.costs() == solution.cost
    assert len(grouper.outputs) == 1
    assert isinstance(grouper.outputs[0], Xor)

    # HalfAdd
    roots = [Bit(), Bit()]
    expected = "00: 00; 01: 01; 10: 01; 11: 10"
    solution = Solver(roots, expected).solve()
    grouper = Grouper(roots, solution.output)
    grouper.simplify()
    assert grouper.costs() == solution.cost
    assert len(grouper.outputs) == 2
