from collections.abc import Sequence
from random import randint
from json import dumps

from numpy import ndarray, roll, stack, apply_along_axis, zeros, base_repr
from pydantic import BaseModel


class CAMetadata(BaseModel):
    cell_states: int
    lattice_width: int
    lattice_configuration_space: int
    time_steps: int
    local_transition_rule: int
    local_transition_neighbourhood_radius: int
    lattice_evolution: list[int]


class CellularAutomata(Sequence):
    def __init__(
        self,
        cell_states: int,
        neighbourhood_radius: int,
        lattice_width: int,
        time_steps: int,
        initial_state: int | None = None,
        transition_rule_number: int | None = None,
    ) -> None:
        super().__init__()

        self.cell_states = cell_states
        self.time_steps = time_steps
        self.neighbourhood_radius = neighbourhood_radius
        local_neighbourhood_size = 2 * neighbourhood_radius + 1
        self.lattice_width = lattice_width

        min_state, max_state = 0, cell_states**lattice_width
        if initial_state is None:
            initial_state = randint(min_state, max_state - 1)
        assert min_state <= initial_state < max_state, "initial state out of bounds"

        min_rule, max_rule = 0, cell_states ** (cell_states**local_neighbourhood_size)
        if transition_rule_number is None:
            transition_rule_number = randint(min_rule, max_rule - 1)

        local_transition_rule = self.create_rule_from_number(
            rule_number=transition_rule_number,
            local_neighbourhood_size=local_neighbourhood_size,
            cell_states=cell_states,
        )

        self.evolution = self.create_spacetime_evolution(
            time_steps=time_steps,
            lattice_width=lattice_width,
            initial_state=initial_state,
            neighbourhood_radius=neighbourhood_radius,
            local_transition_rule=local_transition_rule,
        )

        self.info = CAMetadata(
            cell_states=cell_states,
            lattice_width=lattice_width,
            lattice_configuration_space=max_state,
            time_steps=time_steps,
            local_transition_rule=transition_rule_number,
            local_transition_neighbourhood_radius=neighbourhood_radius,
            lattice_evolution=list(map(self.get_state_number_from_lattice, self)),
        )

    def __len__(self) -> int:
        return self.time_steps

    def __getitem__(self, i: int) -> ndarray:
        return self.evolution[i]

    def __repr__(self) -> str:
        return dumps(self.info.model_dump(mode="json"), indent=2)

    @staticmethod
    def apply_local_transition_rule_to_lattice(
        configuration: ndarray,
        neighbourhood_radius: int,
        local_transition_rule: callable,
    ) -> ndarray:
        local_neighbourhoods = stack(
            [
                roll(configuration, i)
                for i in range(-neighbourhood_radius, neighbourhood_radius + 1)
            ]
        )
        return apply_along_axis(local_transition_rule, 0, local_neighbourhoods)

    @staticmethod
    def create_lattice_from_number(
        state_number: int, lattice_width: int, cell_states: int
    ) -> list[int]:
        s = base_repr(state_number, base=cell_states).zfill(lattice_width)
        return [int(ch) for ch in s]

    def get_state_number_from_lattice(self, lattice: list[int]) -> int:
        return int("".join(str(x) for x in lattice), self.cell_states)

    @staticmethod
    def create_rule_from_number(
        rule_number: int, local_neighbourhood_size: int, cell_states: int
    ) -> callable:
        base_k_str = base_repr(rule_number, base=cell_states).zfill(
            cell_states**local_neighbourhood_size
        )
        outputs = list(map(int, base_k_str[::-1]))

        def local_transition_rule(input_neighbourhood: ndarray) -> int:
            assert input_neighbourhood.shape == (local_neighbourhood_size,)
            lookup_index = 0
            for v in input_neighbourhood:
                lookup_index = lookup_index * cell_states + v
            return outputs[lookup_index]

        return local_transition_rule

    def create_spacetime_evolution(
        self,
        time_steps: int,
        lattice_width: int,
        initial_state: int,
        neighbourhood_radius: int,
        local_transition_rule: callable,
    ) -> ndarray:
        evolution = zeros((time_steps, lattice_width), dtype=int)
        evolution[0, :] = self.create_lattice_from_number(
            initial_state, lattice_width, self.cell_states
        )
        for i in range(1, time_steps):
            evolution[i, :] = self.apply_local_transition_rule_to_lattice(
                evolution[i - 1, :], neighbourhood_radius, local_transition_rule
            )
        return evolution
