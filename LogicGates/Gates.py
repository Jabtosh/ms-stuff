class Gate:
    cost = 0
    fixed_state = None
    mapping = {(0,): 0, (1,): 1}

    def __init__(self, *inputs: 'Gate'):
        self.inputs = inputs
        assert len(self.inputs) == self.expected_number_of_inputs

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

    @property
    def expected_number_of_inputs(self):
        return len(next(iter(self.mapping.keys()))) if self.mapping is not None else 0

    def get_input_recursively(self):
        return {self} | {gate for _input in self.inputs for gate in _input.get_input_recursively()}


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


class Nand(SymmetricGate):
    cost = 1
    mapping = {
        (0, 0): 1,
        (0, 1): 1,
        (1, 0): 1,
        (1, 1): 0
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


class HalfAdder(SymmetricGate):
    cost = 4
    mapping = {
        (0, 0): (0, 0),
        (0, 1): (0, 1),
        (1, 0): (0, 1),
        (1, 1): (1, 0)
    }
