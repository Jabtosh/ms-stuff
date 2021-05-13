from contextlib import contextmanager
from copy import deepcopy
from itertools import product, permutations
from operator import attrgetter

from LogicGates.Gates import Bit, Gate, Nand, And, Or, Xor


@contextmanager
def fix_state(gates, state):
    state_before = [None] * len(gates)
    for n, gate in enumerate(gates):
        state_before[n] = gate.fixed_state
        gate.fixed_state = state[n]
    yield
    for n, gate in enumerate(gates):
        gate.fixed_state = state_before[n]


class Evaluator:
    states = (0, 1)

    def __init__(self, roots: [Bit], outputs: [Gate]):
        self._roots = roots
        self._outputs = outputs

    def compare_with_mapping(self, expected_mapping):
        for state in product(self.states, repeat=len(self._roots)):
            with fix_state(self._roots, state):
                _input = tuple(_root.state for _root in self._roots)
                _output = tuple(gate.state for gate in self._outputs)
                if expected_mapping[_input] != _output:
                    return False

        return True

    def costs(self):
        gates = set()
        for gate in self._outputs:
            gates |= gate.get_input_recursively()
        return sum(gate.cost for gate in gates)


class Grouper(Evaluator):
    gates: [Gate] = [Gate, Nand, And, Or, Xor]
    gates.sort(key=attrgetter("cost"))

    def simplify(self):
        # TODO
        new_circuit = deepcopy(self._outputs)
        for output in new_circuit:
            self._apply_simplifications(output)
        return new_circuit

    def _apply_simplifications(self, circuit_gate: Gate):
        # TODO
        for gate in self.gates:
            costs = circuit_gate.cost
            inputs = circuit_gate.inputs
            if len(set(inputs)) == gate.expected_number_of_inputs and not isinstance(circuit_gate, gate):
                for input_permutation in permutations(circuit_gate.inputs):
                    for in_, out_ in gate.mapping:
                        with fix_state(input_permutation, in_):
                            if circuit_gate.state != out_:
                                break
                    else:
                        return gate
            while costs < gate.cost:
                pass


def main():
    roots = [Bit()]
    outputs = [Nand(roots[0], roots[0])]
    print(outputs)
    g = Grouper(roots, outputs)
    print(g.simplify())


if __name__ == '__main__':
    main()
