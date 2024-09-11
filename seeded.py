
import numpy as np


np.random.seed(0)


def vehicles_in_round(num_rounds, num_vehicles, round):
    s = np.random.poisson(num_vehicles / 2, num_rounds)

    res = s[round - 1]
    if(res > num_vehicles):
        res = num_vehicles
    if(res < 2):
        res = 2
    return res


for i in range(1,11):
    print (vehicles_in_round(11, 5, i))
