import json
from math import factorial as fc


def get_cost_sharing(n, coalitions):
    result = {}
    for player in range(1, n + 1):
        w = 0
        for coop in coalitions.keys():
            if str(player) in coop:
                k = len(coop)
                w += fc(n - k) * fc(k - 1) * (coalitions.get(coop) - coalitions.get(coop.replace(str(player), "")))
        result[player] = w / fc(n)
    return result


with open("coalitions.json", "r") as readFile:
    coalitions_data: dict = json.load(readFile)
players_number, coalitions_from_json = coalitions_data.get("n"), coalitions_data.get("coalitions")
print(coalitions_from_json)
print(get_cost_sharing(players_number, coalitions_from_json))
