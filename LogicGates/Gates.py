from collections import defaultdict
from itertools import product


class Gate:
    states = (0, 1)
    mapping = defaultdict(lambda: 0)
    cost = 0

    __slots__ = ["inputs", "depends_on", "total_cost", "nand_count", "key", "_id"]

    def __init__(self, *inputs: 'Gate'):
        self.inputs = inputs
        self.depends_on = set()
        self.total_cost = self.cost
        self.nand_count = 0
        self.key: (int,) = ()
        self._initialize_from_inputs()

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.total_cost}]_{self._id % 999:03d}()"

    def print(self):
        input_repr = ""
        for _input in self.inputs:
            for line in _input.print().split('\n'):
                input_repr += f"\n\t{line}"
        if input_repr:
            input_repr += "\n"
        return f"{self.__class__.__name__}[{self.total_cost}]_{self._id % 999:03d}({input_repr})"

    def _initialize_from_inputs(self):
        self.key = tuple(self.mapping[tuple(_input.key[i] for _input in self.inputs)]
                         for i in range(len(self.inputs[0].key)))
        self.depends_on = {gate for _input in self.inputs for gate in _input.depends_on} | set(self.inputs)
        self.total_cost = sum(gate.cost for gate in self.depends_on) + self.cost
        self.nand_count = sum(isinstance(gate, Nand) for gate in self.depends_on) + isinstance(self, Nand)
        self._id = sum(value*2**index for index, value in enumerate(self.key))

    @classmethod
    def expected_number_of_inputs(cls):
        return len(next(iter(cls.mapping.keys()))) if cls.mapping is not None else 0

    def get_sub_circuit_recursively(self):
        return {self} | {gate for _input in self.inputs for gate in _input.get_sub_circuit_recursively()}

    get_virtual_sub_circuit_recursively = get_sub_circuit_recursively

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return self.key == other.key and self.total_cost == other.total_cost and self.depends_on == other.depends_on
        return NotImplemented

    def __hash__(self):
        return hash((self.key, self.total_cost))


class Bit(Gate):
    mapping = None

    def __init__(self, index, n_bits):
        self.index = index
        self.n_bits = n_bits
        super().__init__()

    def __repr__(self):
        return f"Bit({self.index}, {self.n_bits})"

    print = __repr__

    @classmethod
    def init_n(cls, n_bits):
        return [cls(index, n_bits) for index in range(n_bits)]

    def _initialize_from_inputs(self):
        self.total_mapping = dict.fromkeys(product(self.states, repeat=self.n_bits))
        for input_vector in self.total_mapping:
            self.total_mapping[input_vector] = int(bool(input_vector[self.n_bits - 1 - self.index]))
        self.key = tuple(self.total_mapping.values())


class SymmetricGate(Gate):
    _instances = {}

    @classmethod
    def new_cached(cls, *inputs):
        key = frozenset(inputs)
        if key not in cls._instances:
            cls._instances[key] = cls(*inputs)
        return cls._instances[key]

    @classmethod
    def reset_cache(cls, gates=None):
        gates = gates or set()
        cls._instances.clear()
        for gate in gates:
            if isinstance(gate, cls):
                key = frozenset(gate.inputs)
                if key not in cls._instances:
                    cls._instances[key] = gate


class Nand(SymmetricGate):
    cost = 1
    mapping = {
        (0, 0): 1,
        (0, 1): 1,
        (1, 0): 1,
        (1, 1): 0
    }


class Inv(SymmetricGate):
    cost = 1
    mapping = {
        (0,): 1,
        (1,): 0
    }


class And(SymmetricGate):
    cost = 2
    mapping = {
        (0, 0): 0,
        (0, 1): 0,
        (1, 0): 0,
        (1, 1): 1
    }


class Or(SymmetricGate):
    cost = 3
    mapping = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 1,
        (1, 1): 1
    }


class Xor(SymmetricGate):
    cost = 4
    mapping = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 1,
        (1, 1): 0
    }


class MultiOut(type):
    def __new__(mcs, name, bases, dct):
        gate_class = super().__new__(mcs, name, bases, dct)
        gate_class.mapping_h = {k: v[0] for k, v in gate_class.mapping.items()}
        gate_class.mapping_l = {k: v[1] for k, v in gate_class.mapping.items()}
        return gate_class


class MultiOutSymmetricGate(SymmetricGate, metaclass=MultiOut):
    _instances = {}
    _selectors = {}
    mapping = {
        (0, 0): (0, 0),
        (0, 1): (0, 1),
        (1, 0): (1, 0),
        (1, 1): (1, 1)
    }

    @classmethod
    def new_cached(cls, *inputs, output_mode='L'):
        instance_key = frozenset(inputs)
        if instance_key not in cls._instances:
            cls._instances[instance_key] = cls(*inputs)
        instance = cls._instances[instance_key]
        selector_key = (output_mode, instance)
        if selector_key not in cls._selectors:
            cls._selectors[selector_key] = OutputSelector(instance, output_mode=output_mode)
        return cls._selectors[selector_key]

    @classmethod
    def reset_cache(cls, gates=None):
        gates = gates or set()
        cls._instances.clear()
        cls._selectors.clear()
        for gate in gates:
            key = frozenset(gate.inputs)
            if isinstance(gate, cls):
                if key not in cls._instances:
                    cls._instances[key] = gate
            elif isinstance(gate, OutputSelector) and isinstance(gate.gate, cls):
                if key not in cls._instances:
                    cls._instances[key] = gate.gate
                selector_key = (gate.output_mode, gate.gate)
                if selector_key not in cls._selectors:
                    cls._selectors[selector_key] = gate


class OutputSelector(Gate, object):
    def __init__(self, gate: MultiOutSymmetricGate, *, output_mode='L'):
        super(object).__init__()
        self.gate = gate
        self.output_mode = output_mode
        index = bool(output_mode == 'L')
        self.total_mapping = {k: v[index] for k, v in self.gate.total_mapping}

    def __repr__(self):
        return f"{self.output_mode}-{repr(self.gate)}"

    @property
    def total_cost(self):
        return self.gate.total_cost

    @property
    def depends_on(self):
        return self.gate.depends_on

    @property
    def inputs(self):
        return self.gate.inputs

    @property
    def mapping(self):
        return self.gate.mapping_l if self.output_mode == 'L' else self.gate.mapping_h

    def expected_number_of_inputs(self):
        return self.gate.expected_number_of_inputs()

    def get_virtual_sub_circuit_recursively(self):
        return {self, type(self.gate).new_cached(*self.inputs, output_mode=('H' if self.output_mode == 'L' else 'L'))} \
               | {gate for _input in self.inputs for gate in _input.get_virtual_sub_circuit_recursively()}

    def get_sub_circuit_recursively(self):
        return {self.gate} | {gate for _input in self.inputs for gate in _input.get_sub_circuit_recursively()}


class HalfAdder(MultiOutSymmetricGate):
    cost = 5
    mapping = {
        (0, 0): (0, 0),
        (0, 1): (0, 1),
        (1, 0): (0, 1),
        (1, 1): (1, 0)
    }


class FullAdder(MultiOutSymmetricGate):
    cost = 25
    mapping = {
        (0, 0, 0): (0, 0),
        (0, 0, 1): (0, 1),
        (0, 1, 0): (0, 1),
        (0, 1, 1): (1, 0),
        (1, 0, 0): (0, 1),
        (1, 0, 1): (1, 0),
        (1, 1, 0): (1, 0),
        (1, 1, 1): (1, 1),
    }
