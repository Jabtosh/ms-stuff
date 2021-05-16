class Gate:
    cost = 0
    fixed_state = None
    mapping = {(0,): 0, (1,): 1}

    def __init__(self, *inputs: 'Gate'):
        self.inputs = inputs
        assert len(self.inputs) == self.expected_number_of_inputs()

    def __repr__(self):
        input_repr = ""
        for _input in self.inputs:
            for line in repr(_input).split('\n'):
                input_repr += f"\n\t{line}"
        if input_repr:
            input_repr += "\n"
        return f"{self.__class__.__name__}_{id(self) % 100000}({input_repr})"

    @property
    def state(self):
        return self.mapping[tuple(_input.state for _input in self.inputs)] if self.fixed_state is None \
            else self.fixed_state

    @classmethod
    def expected_number_of_inputs(cls):
        return len(next(iter(cls.mapping.keys()))) if cls.mapping is not None else 0

    def get_sub_circuit_recursively(self):
        return {self} | {gate for _input in self.inputs for gate in _input.get_sub_circuit_recursively()}

    get_virtual_sub_circuit_recursively = get_sub_circuit_recursively


class Bit(Gate):
    mapping = None
    fixed_state = 0

    def __init__(self):
        super().__init__()


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

    def __repr__(self):
        return f"{self.output_mode}-{repr(self.gate)}"

    @property
    def inputs(self):
        return self.gate.inputs

    @property
    def cost(self):
        return self.gate.cost

    @property
    def mapping(self):
        return self.gate.mapping_l if self.output_mode == 'L' else self.gate.mapping_h

    @property
    def state(self):
        return self.mapping[tuple(_input.state for _input in self.gate.inputs)] if self.gate.fixed_state is None \
            else self.gate.fixed_state[self.output_mode == 'H']

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
