from json import load
from typing import Dict, List
from math import inf

with open("./q.json", "r") as file:
    data: Dict[str, List | Dict] = load(file)
    keys = set(["suppliers", "buyers", "max_supplies", "max_demands", "costs"])
    matrix = list()
    if keys.issubset(data.keys()):
        suppliers = data.get("suppliers", None)
        costs: Dict[str, List[float]] = data.get("costs", None)
        max_supplies: List[float] = data.get("max_supplies", None)
        max_demands: List[float] = data.get("max_demands", None)
        for sup, cost in costs.items():
            idx = suppliers.index(sup)
            cost.append(max_supplies[idx])
            matrix.append(cost)
        max_demands.append(inf)
        matrix.append(max_demands)
        print(matrix)
    else:
        print(f"All keys {keys} must be included")
