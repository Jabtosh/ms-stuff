from contextlib import contextmanager
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
        self.outputs = outputs

    def compare_with_mapping(self, expected_mapping):
        for state in product(self.states, repeat=len(self._roots)):
            with fix_state(self._roots, state):
                _input = tuple(_root.state for _root in self._roots)
                _output = tuple(gate.state for gate in self.outputs)
                if expected_mapping[_input] != _output:
                    return False

        return True

    def costs(self):
        gates = set()
        for gate in self.outputs:
            gates |= gate.get_sub_circuit_recursively()
        return sum(gate.cost for gate in gates)

    def nand_count(self):
        gates = set()
        for gate in self.outputs:
            gates |= gate.get_sub_circuit_recursively()
        return sum(isinstance(gate, Nand) for gate in gates)


class Grouper(Evaluator):
    gates: [Gate] = [Inv, And, Or, Xor]
    gates.sort(key=attrgetter("cost"))

    def __init__(self, roots: [Bit], outputs: [Gate], all_gates=None):
        super().__init__(roots, outputs)
        self._gate_to_replace = None
        if all_gates is None:
            self._all_gates = set()
            for gate in self.outputs:
                self._all_gates |= gate.get_sub_circuit_recursively()
        else:
            self._all_gates = all_gates

    def simplify(self):
        """ Substitute outputs list items in-place for one of cls.gates. """
        for i in range(len(self.outputs)):
            # substitutions strictly including self._outputs[i]
            substitute_gate = self._find_substitute_gate(self.outputs[i])
            while substitute_gate is not None:
                for receiver in self.outputs[i].output_to:
                    receiver.inputs.remove(self.outputs[i])
                    receiver.register_inputs(substitute_gate)
                self.outputs[i] = substitute_gate
                substitute_gate = self._find_substitute_gate(self.outputs[i])

        for i in range(len(self.outputs)):
            # recursive substitutions restricted to new_circuit[i].inputs
            Grouper(self._roots, self.outputs[i].inputs, self._all_gates).simplify()
        return self.outputs

    def _find_substitute_gate(self, circuit_gate: Gate):
        """ Find and return a valid substitute gate from self.gates. """
        self._gate_to_replace = circuit_gate
        for gate in self.gates:
            unique_inputs = list(dict.fromkeys(circuit_gate.inputs))
            substitute_gate = self._find_substitution(gate, unique_inputs, {circuit_gate},
                                                      skip_base_circuit=isinstance(circuit_gate, gate))
            if substitute_gate is not None:
                return substitute_gate
        return None

    def _find_substitution(self, gate: type(Gate), unique_inputs: [Gate], sub_circuit: {Gate}, skip_base_circuit=False):
        """ Return the given gate, correctly wired, if it is a valid substitute. """
        if not skip_base_circuit and sum(c_gate.cost for c_gate in sub_circuit) == gate.cost and \
                all(_output in sub_circuit
                    for c_gate in (g for g in sub_circuit if g is not self._gate_to_replace)
                    for _output in c_gate.output_to):
            circuit_to_gate_mapping = self._get_input_mapping(gate, unique_inputs)
            if circuit_to_gate_mapping is not None:
                for _input in unique_inputs:
                    _input.output_to.clear()
                return gate(*[unique_inputs[index] for index in circuit_to_gate_mapping])

        for expanded_input in [_input for _input in unique_inputs if _input.inputs]:
            _sub_circuit = sub_circuit | {expanded_input}
            if sum(c_gate.cost for c_gate in _sub_circuit) <= gate.cost:
                _inputs = [_input for _input in unique_inputs if _input is not expanded_input]
                _inputs.extend(expanded_input.inputs)
                _unique_inputs = list(dict.fromkeys(_inputs))

                substitute_gate = self._find_substitution(gate, _unique_inputs, _sub_circuit)
                if substitute_gate is not None:
                    return substitute_gate

        return None

    def _get_input_mapping(self, gate: type(Gate), unique_inputs: [Gate]) -> [int]:
        """ Return the circuit to gate mapping [int]. E.g [2, 2, 1] implies gate(unique_inputs[2], u_i[2], u_i[1]). """
        # for the gate to fit, sub-circuit inputs must be <= gate inputs
        # as the gate should be optimal, sub-circuit costs must be >= gate costs
        n_inputs = gate.expected_number_of_inputs()
        if len(unique_inputs) <= n_inputs:
            sub_circuit_mapping = self._get_sub_circuit_mapping(unique_inputs)
            columns = list(range(len(unique_inputs))) * (1 + n_inputs - len(unique_inputs))
            for column_permutation in set(permutations(columns, n_inputs)):
                if self._permute_mapping(sub_circuit_mapping, column_permutation) == gate.mapping:
                    return column_permutation
        return None

    def _get_sub_circuit_mapping(self, unique_inputs: [Gate]) -> {tuple: tuple}:
        """ Get the state mapping of the current sub-circuit. """
        sub_circuit_mapping = {}
        for input_state in product(self.states, repeat=len(unique_inputs)):
            with fix_state(unique_inputs, input_state):
                sub_circuit_mapping[input_state] = (self._gate_to_replace.state,)
        return sub_circuit_mapping

    @staticmethod
    def _permute_mapping(sub_circuit_mapping: {tuple: tuple}, column_permutation: [int]):
        """ Return new sub_circuit_mapping with permuted columns. Convention: if len(output)==1 it gets unpacked. """
        return {
            tuple(_in[index] for index in column_permutation): (_out[0] if len(_out) == 1 else _out)
            for _in, _out in sub_circuit_mapping.items()
        }


def main():
    from LogicGates.Solver import Solver
    roots = [Bit(), Bit()]
    expected = "00: 00; 01: 01; 10: 01; 11: 10"
    solution = Solver(roots, expected).solve()
    print(solution.output)
    grouper = Grouper(roots, solution.output)
    print(grouper.simplify())


if __name__ == '__main__':
    main()
