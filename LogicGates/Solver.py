from collections import namedtuple
from itertools import product
from operator import attrgetter

from LogicGates.Evaluation import Evaluator
from LogicGates.Gates import Bit, Nand, HalfAdder

Solution = namedtuple("Solution", ["output", "cost"])
_solution_sep = ',\n'
Solution.__repr__ = lambda self: f"Solution ({self.cost} Nands):\n{_solution_sep.join(map(repr, self.output))}\n"


class Solver:
    gate_types = {Nand}

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
            for line in representation.replace(" ", "").replace("\n", "").split(";")
        }

    def solve(self):
        mapping = self._expected_mapping
        gates = set()
        new_gates = {*self._roots}
        valid_solutions = self._get_solutions_from_new_gates(gates, new_gates, mapping)
        gates |= new_gates
        while not valid_solutions:
            # avoid redundancy, by only adding gates with at least one new gate as input (use nand symmetry)
            new_gates = self._get_new_gates(gates, new_gates)
            valid_solutions = self._get_solutions_from_new_gates(gates, new_gates, mapping)
            gates |= new_gates
        self.solution = min(valid_solutions, key=attrgetter("cost"))
        return self.solution

    def _get_new_gates(self, gates, new_gates):
        return set(gate_type.new_cached(in1, in2) for gate_type in self.gate_types
                   for in1, in2 in product(new_gates, new_gates | gates))

    def _get_solutions_from_new_gates(self, gates, new_gates, mapping):
        valid_solutions = []
        for new_gate in new_gates:
            solution = self._check_gate_combinations(gates, new_gate, mapping)
            if solution is not None:
                valid_solutions.append(solution)
        return valid_solutions

    def _check_gate_combinations(self, gates, new_gate, mapping):
        valid_combinations = set(product(gates, repeat=(self._expect_n_outputs - 1)))
        for output_template in valid_combinations:
            for position in range(len(output_template) + 1):
                output_attempt = list(output_template)
                output_attempt.insert(position, new_gate)
                try:
                    if Evaluator(self._roots, output_attempt).compare_with_mapping(mapping):
                        return Solution(output_attempt, Evaluator(self._roots, output_attempt).costs())
                except KeyError as e:
                    print(f"KeyError {e} with {output_attempt}")
        return None


class HSolver(Solver):
    """ Uses a heuristic to solve more efficiently. """

    def solve(self):
        expect_n_outputs = self._expect_n_outputs
        self._expect_n_outputs = 1
        new_gates = {*self._roots}
        output = []
        for output_index in range(expect_n_outputs):
            gates = set()
            mapping = self._select_mapping_column(output_index)
            valid_solutions = self._get_solutions_from_new_gates(gates, new_gates, mapping)
            gates |= new_gates
            while not valid_solutions:
                # avoid redundancy, by only adding gates with at least one new gate as input (use nand symmetry)
                new_gates = self._get_new_gates(gates, new_gates)
                valid_solutions = self._get_solutions_from_new_gates(gates, new_gates, mapping)
                gates |= new_gates
            bit_solution = min(valid_solutions, key=attrgetter("cost")).output
            output.append(bit_solution[0])
            new_gates = bit_solution[0].get_virtual_sub_circuit_recursively()
            self._reset_cache(new_gates)
        return Solution(output, Evaluator(self._roots, output).costs())

    @staticmethod
    def _reset_cache(new_gates):
        Nand.reset_cache(new_gates)

    def _get_new_gates(self, gates, new_gates):
        return set(Nand.new_cached(in1, in2) for in1, in2 in product(new_gates, new_gates | gates))

    def _select_mapping_column(self, index):
        return {key: (value[index],) for key, value in self._expected_mapping.items()}


class HAHSolver(HSolver):
    """ Uses a heuristic to solve more efficiently and implements half adders in addition to nands. """

    def _get_new_gates(self, gates, new_gates):
        return set(Nand.new_cached(in1, in2) for in1, in2 in product(new_gates, new_gates | gates)) | \
               set(HalfAdder.new_cached(in1, in2, output_mode=mode) for mode in ['L', 'H']
                   for in1, in2 in product(new_gates, new_gates | gates))

    @staticmethod
    def _reset_cache(new_gates):
        Nand.reset_cache(new_gates)
        HalfAdder.reset_cache(new_gates)


def main():
    print("Id")
    roots = [Bit()]
    expected = "0: 0; 1: 1"
    print(Solver(roots, expected).solve())

    print("Inv")
    expected = "0: 1; 1: 0"
    print(Solver(roots, expected).solve())

    print("AND")
    roots = [Bit(), Bit()]
    expected = "00: 0; 01: 0; 10: 0; 11: 1"
    print(Solver(roots, expected).solve())

    print("OR")
    roots = [Bit(), Bit()]
    expected = "00: 0; 01: 1; 10: 1; 11: 1"
    print(Solver(roots, expected).solve())

    print("XOR")
    roots = [Bit(), Bit()]
    expected = "00: 0; 01: 1; 10: 1; 11: 0"
    print(Solver(roots, expected).solve())

    print("Half Adder")
    roots = [Bit(), Bit()]
    expected = "00: 00; 01: 01; 10: 01; 11: 10"
    print(HSolver(roots, expected).solve())

    print("Full Adder")
    roots = [Bit(), Bit(), Bit()]
    expected = "000: 00; 001: 01; 010: 01; 011: 10; 100: 01; 101: 10; 110: 10; 111: 11"
    print(HSolver(roots, expected).solve())


if __name__ == "__main__":
    main()
