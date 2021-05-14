from contextlib import contextmanager
from copy import deepcopy
from itertools import product, permutations
from operator import attrgetter

from LogicGates.Gates import Bit, Gate, Nand, And, Or, Xor, Inv


@contextmanager
def fix_state(gates, state):
    state_before = [gate.fixed_state for gate in gates]
    for n, gate in enumerate(gates):
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

    def nand_count(self):
        gates = set()
        for gate in self._outputs:
            gates |= gate.get_input_recursively()
        return sum(isinstance(gate, Nand) for gate in gates)


class Grouper(Evaluator):
    gates: [Gate] = [Inv, Nand, And, Or, Xor]
    gates.sort(key=attrgetter("cost"))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.new_circuit = list(deepcopy(self._outputs))

    def simplify(self):
        for i in range(len(self.new_circuit)):
            # substitutions strictly including self._outputs[i]
            substitute_gate = self._find_substitute_gate(self.new_circuit[i])
            while substitute_gate is not None:
                self.new_circuit[i] = substitute_gate
                substitute_gate = self._find_substitute_gate(self.new_circuit[i])

            # recursive substitutions restricted to new_circuit[i].inputs
            evaluator = Evaluator(self._roots, self.new_circuit[i].inputs)
            substitute_inputs = Grouper(self._roots, self.new_circuit[i].inputs).simplify()
            substitute_evaluator = Evaluator(self._roots, substitute_inputs)
            if substitute_evaluator.costs() < evaluator.costs() \
                    or substitute_evaluator.nand_count() < evaluator.nand_count():
                self.new_circuit[i].inputs = substitute_inputs
        return self.new_circuit

    def _find_substitute_gate(self, circuit_gate: Gate):
        for gate in self.gates:
            _inputs = circuit_gate.inputs
            unique_inputs = list(dict.fromkeys(_inputs))
            skip_base_circuit = isinstance(circuit_gate, gate)
            substitute_gate = self._find_substitution(gate, unique_inputs, {circuit_gate},
                                                      skip_base_circuit=skip_base_circuit)
            if substitute_gate is not None:
                return substitute_gate
        return None

    def _find_substitution(self, gate, unique_inputs: list, sub_circuit: set, skip_base_circuit=False):
        if not skip_base_circuit and sum(c_gate.cost for c_gate in sub_circuit) == gate.cost:
            circuit_to_gate_mapping = self._get_input_mapping(gate, unique_inputs)
            if circuit_to_gate_mapping is not None:
                return gate(*[unique_inputs[index] for index in circuit_to_gate_mapping])

        for expanded_input in [_input for _input in unique_inputs if _input.inputs]:
            _sub_circuit = sub_circuit | {c_gate for c_gate in expanded_input.inputs}
            if sum(c_gate.cost for c_gate in _sub_circuit) <= gate.cost:
                _inputs = [c_gate for c_gate in unique_inputs if c_gate is not expanded_input]
                _inputs.extend(expanded_input.inputs)
                _unique_inputs = list(dict.fromkeys(_inputs))

                circuit_to_gate_mapping = self._find_substitution(gate, _unique_inputs, _sub_circuit)
                if circuit_to_gate_mapping is not None:
                    return gate(*[_unique_inputs[index] for index in circuit_to_gate_mapping])

        return None

    def _get_input_mapping(self, gate, unique_inputs) -> [int]:
        """ Return the circuit to gate mapping [int]. E.g [2, 2, 1] implies gate(unique_inputs[2], u_i[2], u_i[1]). """
        # for the gate to fit, sub-circuit inputs must be <= gate inputs
        # as the gate should be optimal, sub-circuit costs must be >= gate costs
        n_inputs = gate.expected_number_of_inputs()
        if len(unique_inputs) <= n_inputs:
            sub_circuit_mapping = self._get_sub_circuit_mapping(unique_inputs)
            columns = list(range(len(unique_inputs))) * (1 + n_inputs - len(unique_inputs))
            # TODO: cast permutations iterator to set?
            for column_permutation in permutations(columns, n_inputs):
                if self._permute_mapping(sub_circuit_mapping, column_permutation) == gate.mapping:
                    return column_permutation
        return None

    def _get_sub_circuit_mapping(self, unique_inputs: [Gate]) -> {tuple: tuple}:
        sub_circuit_mapping = {}
        for input_state in product(self.states, repeat=len(unique_inputs)):
            with fix_state(unique_inputs, input_state):
                sub_circuit_mapping[input_state] = tuple(gate.state for gate in self.new_circuit)
        return sub_circuit_mapping

    @staticmethod
    def _permute_mapping(sub_circuit_mapping: {tuple: tuple}, column_permutation: [int]):
        """ Return new sub_circuit_mapping with permuted columns. Convention: if len(output)==1 it gets unpacked. """
        return {
            tuple(_in[index] for index in column_permutation): (_out[0] if len(_out) == 1 else _out)
            for _in, _out in sub_circuit_mapping.items()
        }


def main():
    roots = [Bit()]
    outputs = [Nand(roots[0], roots[0])]
    print(outputs)
    g = Grouper(roots, outputs)
    print(g.simplify())

    roots = [Bit(), Bit()]
    _nand = Nand(roots[0], roots[1])
    outputs = [Nand(_nand, _nand)]
    print(outputs)
    g = Grouper(roots, outputs)
    print(g.simplify())


if __name__ == '__main__':
    main()
