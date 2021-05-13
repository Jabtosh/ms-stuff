from collections import namedtuple
from itertools import permutations
from operator import attrgetter

from LogicGates.Evaluation import Evaluator
from LogicGates.Gates import Bit, Nand


Solution = namedtuple("Solution", ["output", "cost"])
_solution_sep = ',\n'
Solution.__repr__ = lambda self: f"Solution ({self.cost} Nands):\n{_solution_sep.join(map(repr, self.output))}\n"


class Solver:

    def __init__(self, roots: [Bit], expected_transformation: str):
        self._roots = roots
        self._expected_mapping = self.parse_expectation(expected_transformation)
        self.solution = None
        try:
            self._expect_n_outputs = len(next(iter(self._expected_mapping.values())))
        except TypeError:
            self._expect_n_outputs = 1

    @staticmethod
    def parse_expectation(representation: str):
        return {
            tuple(int(x) for x in line.split(":")[0]): tuple(int(x) for x in line.split(":")[1])
            for line in representation.replace(" ", "").replace("\n", "").split(";")}

    def solve(self):
        gates = [*self._roots]
        valid_solutions = []
        solution = self._check_gate_combinations_full(gates)
        if solution is not None:
            valid_solutions.append(solution)
        while not valid_solutions:
            new_gates = (Nand.new_cached(in1, in2) for in1, in2 in {(in1_, in2_) for in1_ in gates for in2_ in gates})
            valid_solutions = []
            for new_gate in new_gates:
                if new_gate not in gates:
                    solution = self._check_gate_combinations(gates, new_gate)
                    gates.append(new_gate)
                    if solution is not None:
                        valid_solutions.append(solution)
        self.solution = min(valid_solutions, key=attrgetter("cost"))
        print(self.solution)

    def _check_gate_combinations(self, gates, new_gate):
        valid_combinations = set(permutations(gates * (self._expect_n_outputs - 1), (self._expect_n_outputs - 1)))
        for output_template in valid_combinations:
            for position in range(len(output_template) + 1):
                output_attempt = list(output_template)
                output_attempt.insert(position, new_gate)
                if Evaluator(self._roots, output_attempt).compare_with_mapping(self._expected_mapping):
                    return Solution(output_attempt, Evaluator(self._roots, output_attempt).costs())
        return None

    def _check_gate_combinations_full(self, gates):
        valid_combinations = set(permutations(gates * self._expect_n_outputs, self._expect_n_outputs))
        for output_attempt in valid_combinations:
            if Evaluator(self._roots, output_attempt).compare_with_mapping(self._expected_mapping):
                return Solution(output_attempt, Evaluator(self._roots, output_attempt).costs())
        return None


def main():
    print("Id")
    roots = [Bit()]
    expected = "0: 0; 1: 1"
    Solver(roots, expected).solve()

    print("Inv")
    expected = "0: 1; 1: 0"
    Solver(roots, expected).solve()

    print("AND")
    roots = [Bit(), Bit()]
    expected = "00: 0; 01: 0; 10: 0; 11: 1"
    Solver(roots, expected).solve()

    print("OR")
    roots = [Bit(), Bit()]
    expected = "00: 0; 01: 1; 10: 1; 11: 1"
    Solver(roots, expected).solve()

    print("XOR")
    roots = [Bit(), Bit()]
    expected = "00: 0; 01: 1; 10: 1; 11: 0"
    Solver(roots, expected).solve()

    print("ADD")
    roots = [Bit(), Bit()]
    expected = "00: 00; 01: 01; 10: 01; 11: 10"
    Solver(roots, expected).solve()


if __name__ == "__main__":
    main()
