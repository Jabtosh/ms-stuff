from collections import defaultdict
from itertools import product, chain
from operator import attrgetter, methodcaller

from LogicGates.Gates import Bit, Nand, HalfAdder, Gate


class Solution:
    def __init__(self, output):
        self.output = output
        self.cost = Gate(*output).total_cost

    def __repr__(self):
        output_repr = ',\n'.join(map(methodcaller("print"), self.output))
        return f"Solution ({self.cost} Nands):\n{output_repr}\n"


def iter_values(gate_dict):
    return chain.from_iterable(gate_dict.values())


def item_from_iterable(non_empty_iterable):
    """ Fastest way in Python. """
    for item in non_empty_iterable:
        break
    return item


class Solver:
    gate_types = {Nand}

    def __init__(self, expected_transformation: str):
        self._expected_mapping = self.parse_expectation(expected_transformation)
        self._expected_key = tuple(self._expected_mapping.values())
        self._n_inputs = len(next(iter(self._expected_mapping.keys())))
        self._n_outputs = len(next(iter(self._expected_mapping.values())))
        self._key_len = len(self._expected_key)
        self._roots = Bit.init_n(self._n_inputs)
        self._key_by_column = [tuple(self._expected_key[row][col] for row in range(self._key_len))
                               for col in range(self._n_outputs)]
        self.solution = None

    @staticmethod
    def parse_expectation(representation: str):
        return {
            tuple(int(x) for x in line.split(":")[0]): tuple(int(x) for x in line.split(":")[1])
            for line in representation.replace(" ", "").replace("\n", "").split(";")
        }

    def solve(self):
        gates = {}
        new_gates = {root.key: {root} for root in self._roots}
        valid_solutions = self._get_solutions_from_new_gates(gates, new_gates)
        gates.update(new_gates)
        while not valid_solutions:
            new_gates = self._get_new_gates(gates, new_gates)
            valid_solutions = self._get_solutions_from_new_gates(gates, new_gates)
            gates.update(new_gates)
        self.solution = min(valid_solutions, key=attrgetter("cost"))
        return self.solution

    def _get_new_gates(self, gates: {tuple: [Gate]}, new_gates: {tuple: [Gate]}):
        # Avoid redundancy, by only adding gates with at least one new gate as input (use nand symmetry)
        candidates = set(gate_type(item_from_iterable(in1), item_from_iterable(in2)) for gate_type in self.gate_types
                         for in1, in2 in product(new_gates.values(), gates.values()))
        # For every key not already in gates: return the lowest total_cost gates with it
        new_gates = defaultdict(set)
        for candidate in filter(lambda g: g.key not in gates.keys(), candidates):
            new_gates[candidate.key].add(candidate)
        # Filter out higher cost gate alternatives
        new_gates = {key: self._get_lowest_total_cost_elements(gate_group) for key, gate_group in new_gates.items()}
        return new_gates

    @staticmethod
    def _get_lowest_total_cost_elements(gate_group):
        minimum = min(map(attrgetter("total_cost"), gate_group))
        return {gate for gate in gate_group if gate.total_cost == minimum}

    def _get_solutions_from_new_gates(self, gates: {tuple: [Gate]}, new_gates: {tuple: [Gate]}):
        output = dict.fromkeys(self._key_by_column)
        for sought_key in self._key_by_column:
            if sought_key in gates.keys():
                output[sought_key] = gates[sought_key]
            elif sought_key in new_gates.keys():
                output[sought_key] = new_gates[sought_key]
            else:
                break
        else:
            return [Solution(*viable_solution) for viable_solution in product(output.values())]
        return []


class CSolver(Solver):
    """ Solves output-bit-wise. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sought_keys = dict.fromkeys(self._key_by_column)

    def solve(self):
        gates = {}
        new_gates = {root.key: {root} for root in self._roots}
        self._add_partial_solutions(new_gates)
        gates.update(new_gates)
        while None in self._sought_keys.values():
            new_gates = self._get_new_gates(gates, new_gates)
            self._add_partial_solutions(new_gates)
            gates.update(new_gates)
        self.solution = Solution(list(self._sought_keys.values()))
        return self.solution

    def _get_new_gates(self, gates: {tuple: [Gate]}, new_gates: {tuple: [Gate]}):
        # avoid redundancy, by only adding gates with at least one new gate as input (use nand symmetry)
        candidates = set(gate_type(in1, in2) for gate_type in self.gate_types
                         for in1, in2 in set(product(iter_values(new_gates), iter_values(gates))))
        # For every key not already in gates: return the lowest total_cost gates with it
        new_gates = defaultdict(set)
        for candidate in filter(lambda g: g.key not in gates.keys(), candidates):
            new_gates[candidate.key].add(candidate)
        new_gates = {key: self._get_lowest_total_cost_elements(gate_group) for key, gate_group in new_gates.items()}
        return new_gates

    def _add_partial_solutions(self, new_gates: {tuple: [Gate]}):
        for key, gate in {
            new_gate.key: new_gate
            for new_gate in iter_values(new_gates)
            if new_gate.key in self._sought_keys
        }.items():
            if self._sought_keys[key] is None or self._sought_keys[key].total_cost > gate.total_cost:
                self._sought_keys[key] = gate


class FullSolver(Solver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._key_by_column = [tuple(self._expected_key[row][col] for row in range(self._key_len))
                               for col in range(self._n_outputs)]

    def _get_new_gates(self, gates: {tuple: [Gate]}, new_gates: {tuple: [Gate]}):
        # Avoid redundancy, by only adding gates with at least one new gate as input (use nand symmetry)
        candidates = set(gate_type(item_from_iterable(in1), item_from_iterable(in2)) for gate_type in self.gate_types
                         for in1, in2 in product(new_gates.values(), gates.values()))
        # For every key not already in gates: return the lowest total_cost gates with it
        new_gates = defaultdict(set)
        for candidate in filter(lambda g: g.key not in gates.keys(), candidates):
            new_gates[candidate.key].add(candidate)
        new_gates = {key: self._get_lowest_total_cost_elements(gate_group) for key, gate_group in new_gates.items()}
        return new_gates


class HAHSolver(Solver):
    """ Uses a heuristic to solve more efficiently and implements half adders in addition to nands. """

    def _get_new_gates(self, gates, new_gates):
        # avoid redundancy, by only adding gates with at least one new gate as input (use nand symmetry)
        candidates = set(Nand.new_cached(in1, in2) for in1, in2
                         in set(product(new_gates.values(), chain(new_gates.values(), gates.values())))) | \
                     set(HalfAdder.new_cached(in1, in2, output_mode=mode) for mode in ['L', 'H']
                         for in1, in2 in set(product(new_gates.values(), chain(new_gates.values(), gates.values()))))
        # For every key not already in gates: return the lowest total_cost gate with it
        return {
            candidate.key: candidate
            for candidate in sorted(filter(lambda g: g.key not in gates.keys(), candidates),
                                    key=lambda g: g.total_cost, reverse=True)
        }

    @staticmethod
    def _reset_cache(new_gates):
        Nand.reset_cache(new_gates)
        HalfAdder.reset_cache(new_gates)


def main():
    print("Id")
    expected = "0: 0; 1: 1"
    print(Solver(expected).solve())

    print("Inv")
    expected = "0: 1; 1: 0"
    print(Solver(expected).solve())

    print("AND")
    expected = "00: 0; 01: 0; 10: 0; 11: 1"
    print(Solver(expected).solve())

    print("OR")
    expected = "00: 0; 01: 1; 10: 1; 11: 1"
    print(Solver(expected).solve())

    print("XOR")
    expected = "00: 0; 01: 1; 10: 1; 11: 0"
    print(Solver(expected).solve())

    print("Half Adder")
    expected = "00: 00; 01: 01; 10: 01; 11: 10"
    print(Solver(expected).solve())

    print("Full Adder")
    expected = "000: 00; 001: 01; 010: 01; 011: 10; 100: 01; 101: 10; 110: 10; 111: 11"
    print(Solver(expected).solve())


if __name__ == "__main__":
    main()
