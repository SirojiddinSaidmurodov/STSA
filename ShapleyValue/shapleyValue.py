import json
from math import factorial

with open("coalitions.json", "r") as readFile:
    coalitions_data: dict = json.load(readFile)
n = coalitions_data.get("n")
coalitions: dict = coalitions_data.get("coalitions")
print(coalitions)
result = []
for player in range(1, n + 1):
    w = 0
    for coalition in coalitions.keys():
        if str(player) in coalition:
            k = len(coalition)
            w += factorial(n - k) * factorial(k - 1) * (
                    coalitions.get(coalition) - coalitions.get(coalition.replace(str(player), "")))
    result.append(w / factorial(n))
print(result)
