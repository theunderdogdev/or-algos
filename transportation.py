from typing import List, Self, Dict
from copy import deepcopy
from numpy import argmin, unravel_index, array, diff, inf, isinf, where
from math import isinf as misinf
from json import load
from os.path import exists


class Transportation:
    def __diff_arr(self, arr: List) -> List:
        if len(arr) > 1:
            return diff(where(isinf(arr), 0.0, arr))
        else:
            return arr

    def __init__(self) -> None:
        self.matrix: List[List[int | float]] | None = None
        self.is_balanced: bool = False
        self.cost = 0
        self.__verbose: bool = False

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
        return all([row[-1] == 0 or misinf(row[-1]) for row in self.matrix]) and all(
            [demand == 0 or misinf(demand) for demand in self.matrix[-1]]
        )

    def __north_west(self) -> None:
        if self.__verbose:
            print(array(self.matrix))
        i, j = 0, 0
        while (
            not self.__is_exhausted()
            and i < len(self.matrix) - 1
            and j < len(self.matrix[0]) - 1
        ):
            factor = min(self.matrix[i][-1], self.matrix[-1][j])
            if self.__verbose:
                print(f"({i}, {j}), Factor: {factor}, cost: {self.matrix[i][j]}")

            self.cost = self.cost + self.matrix[i][j] * factor
            self.matrix[i][-1] -= factor
            self.matrix[-1][j] -= factor
            if self.matrix[i][-1] == 0:
                i += 1
            if self.matrix[-1][j] == 0:
                j += 1

    def __least_cost(self):
        if self.__verbose:
            print(array(self.matrix))
        arr = array(self.matrix)
        arr = arr[0 : arr.shape[0] - 1, 0 : arr.shape[1] - 1]
        while not self.__is_exhausted():
            i, j = unravel_index(argmin(arr), arr.shape)
            factor = min(self.matrix[i][-1], self.matrix[-1][j])
            if self.__verbose:
                print(f"({i}, {j}), Factor: {factor}, cost: {self.matrix[i][j]}")
            self.cost = self.cost + self.matrix[i][j] * factor
            self.matrix[i][-1] -= factor
            self.matrix[-1][j] -= factor
            arr[i][j] = inf
            if self.matrix[i][-1] == 0:
                arr[i, :] = inf
            if self.matrix[-1][j] == 0:
                arr[:, j] = inf

    def __vojels_approx(self):
        if self.__verbose:
            print(array(self.matrix))
        arr = array(self.matrix)
        arr = arr[0 : arr.shape[0] - 1, 0 : arr.shape[1] - 1]
        row_pens = array([0.0 for _ in range(arr.shape[0])])
        col_pens = array([0.0 for _ in range(arr.shape[1])])
        while not self.__is_exhausted():
            for r in range(arr.shape[0]):
                mins_row = sorted(set(arr[r, :]))[:2]
                row_pens[r] = abs(self.__diff_arr(mins_row)[0])
            for c in range(arr.shape[1]):
                mins_col = sorted(set(arr[:, c]))[:2]
                col_pens[c] = abs(self.__diff_arr(mins_col)[0])
            i, j = 0, 0
            rev_col = where(isinf(col_pens), -inf, col_pens)
            rev_row = where(isinf(row_pens), -inf, row_pens)
            if max(rev_row) < max(rev_col):
                j = unravel_index(
                    where(isinf(col_pens), -inf, col_pens).argmax(), col_pens.shape
                )
                i = unravel_index(argmin(arr[:, j]), arr[:, j].shape)
                i, j = i[0], j[0]
            else:
                i = unravel_index(
                    where(isinf(row_pens), -inf, row_pens).argmax(), row_pens.shape
                )
                j = unravel_index(argmin(arr[i, :][0]), arr[i, :][0].shape)
                i, j = i[0], j[0]
            factor = min(self.matrix[i][-1], self.matrix[-1][j])
            if self.__verbose:
                print(f"({i}, {j}), Factor: {factor}, cost: {self.matrix[i][j]}")
            self.cost = self.cost + self.matrix[i][j] * factor
            self.matrix[i][-1] -= factor
            self.matrix[-1][j] -= factor
            if self.matrix[i][-1] == 0:
                arr[i, :] = inf
                row_pens[i] = inf
            if self.matrix[-1][j] == 0:
                arr[:, j] = inf
                col_pens[j] = inf

    def solve(self, method: str = "NW", verbose: bool = False) -> Self:
        self.__verbose = verbose
        self.cost = 0
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
    solver = Transportation()
    print(solver.input_file("./q.json").solve("NW").cost)
    print(solver.input_file("./q.json").solve("LC").cost)
    print(solver.input_file("./q.json").solve("VA").cost)
    print("=============================================")
    print(solver.input_file("./q2.json").solve("NW").cost)
    print(solver.input_file("./q2.json").solve("LC").cost)
    print(solver.input_file("./q2.json").solve("VA").cost)
