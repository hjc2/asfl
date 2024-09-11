
import numpy as np

# np.random.seed(0)

def vehicles_in_round(num_rounds, num_vehicles, round):
    np.random.seed(0)

    s = np.random.poisson(num_vehicles / 2, num_rounds)

    res = s[round - 1]
    if(res > num_vehicles):
        res = num_vehicles
    if(res < 2):
        res = 2
    return res



def old_in_round(num_rounds, num_vehicles, round):
    
    a = max((int( (num_vehicles * round) / num_rounds ) + 1), 2)

    return min(a, num_vehicles)

# for i in range(1,6):
#     print (vehicles_in_round(5, 5, i))

# print(vehicles_in_round(8, 8, 2))