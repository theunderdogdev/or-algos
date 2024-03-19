from typing import List, Self, Dict
from copy import deepcopy
from numpy import argmin, unravel_index, array
from math import inf, isinf
from json import load
from os.path import exists


class Transportation:
    def __init__(self) -> None:
        self.matrix: List[List[int | float]] | None = None
        self.is_balanced: bool = False
        self.cost = 0

    def input_file(self, json_path: str) -> Self:
        if not exists(json_path):
            print(f"Please make sure {json_path} exists")
        with open(json_path, "r") as file:
            data: Dict[str, List | Dict] = load(file)
            keys = set(["suppliers", "buyers", "max_supplies", "max_demands", "costs"])
            matrix = list()
            if keys.issubset(data.keys()):
                suppliers = data.get("suppliers", None)
                buyers = data.get("buyers", None)
                costs: Dict[str, List[float]] = data.get("costs", None)
                max_supplies: List[float] = data.get("max_supplies", None)
                max_demands: List[float] = data.get("max_demands", None)
                self.is_balanced = sum(max_supplies) == sum(max_demands)
                if not self.is_balanced and len(max_demands) != len(buyers):
                    print(
                        "Size of Supplies doesn't match the length of cost matrix"
                        if self.is_balanced
                        else "The demands and supplies aren't balanced"
                    )
                    return self
                for sup, cost in costs.items():
                    idx = suppliers.index(sup)
                    cost.append(max_supplies[idx])
                    matrix.append(cost)
                max_demands.append(inf)
                matrix.append(max_demands)
                self.matrix = deepcopy(matrix)
            else:
                print(f"All keys {keys} must be included")
        return self
    def input(
        self,
        cost_matrix: List[List[int | float]],
        supplies: List[int | float],
        demands: List[int | float],
    ) -> Self:
        self.is_balanced = sum(supplies) == sum(demands)

        if self.is_balanced and len(demands) == len(cost_matrix[0]) != 0:
            for i, row in enumerate(cost_matrix):
                row.append(supplies[i])
            demands.append(inf)
            cost_matrix.append(demands)
            self.matrix = deepcopy(cost_matrix)
        else:
            print(
                "Size of Supplies doesn't match the length of cost matrix"
                if self.is_balanced
                else "The demands and supplies aren't balanced"
            )
        return self

    def __is_exhausted(self):
        return all([row[-1] == 0 or isinf(row[-1]) for row in self.matrix]) and all(
            [demand == 0 or isinf(demand) for demand in self.matrix[-1]]
        )

    def __north_west(self) -> None:
        _i, _j = 0, 0
        while (
            not self.__is_exhausted()
            and _i < len(self.matrix) - 1
            and _j < len(self.matrix[0]) - 1
        ):
            factor = min(self.matrix[_i][-1], self.matrix[-1][_j])
            self.cost = self.cost + self.matrix[_i][_j] * factor
            self.matrix[_i][-1] -= factor
            self.matrix[-1][_j] -= factor
            if self.matrix[_i][-1] == 0:
                _i += 1
            if self.matrix[-1][_j] == 0:
                _j += 1

    def __least_cost(self):
        arr = array(self.matrix)
        print(self.matrix)
        arr = arr[0 : arr.shape[0] - 1, 0 : arr.shape[1] - 1]
        while not self.__is_exhausted():
            i, j = unravel_index(argmin(arr), arr.shape)
            factor = min(self.matrix[i][-1], self.matrix[-1][j])
            self.cost = self.cost + self.matrix[i][j] * factor
            self.matrix[i][-1] -= factor
            self.matrix[-1][j] -= factor
            arr[i][j] = inf
            if self.matrix[i][-1] == 0:
                arr[i, :] = inf
            if self.matrix[-1][j] == 0:
                arr[:, j] = inf

    def __vojels_approx(self):
        pass

    def solve(self, method: str = "NW") -> Self:
        if self.matrix is not None:
            print("Calling solve, method: ", method)
            match method:
                case "NW":
                    self.__north_west()
                case "LC":
                    self.__least_cost()
                case "VA":
                    self.__vojels_approx()
                case _:
                    print(f"No {method} method found")
        return self


if __name__ == "__main__":
    # transp = (
    #     Transportation()
    #     .input(
    #         [
    #             [
    #                 16,
    #                 18,
    #                 21,
    #                 22,
    #             ],
    #             [
    #                 17,
    #                 19,
    #                 14,
    #                 13,
    #             ],
    #             [32, 11, 15, 10],
    #         ],
    #         [150, 160, 90],
    #         [140, 120, 90, 50],
    #     )
    #     .solve(method="NW")
    # )

    # print(transp.cost)
    print(Transportation().input_file('./q.json').solve(method="NW").cost)
