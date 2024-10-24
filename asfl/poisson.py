
import numpy as np

# THIS FILE DETERMINES HOW MANY VEHICLES ARE INCLUDED IN THE TRAINING ROUND

def vehicles_in_round(num_rounds, num_vehicles, round, fraction=2):
    '''Determines the # of vehicles in the round'''
    np.random.seed(0)

    s = np.random.poisson(num_vehicles / fraction, num_rounds)

    res = s[round - 1]
    if(res > num_vehicles):
        res = num_vehicles
    if(res < 2):
        res = 2
    return res

def old_in_round(num_rounds, num_vehicles, round):
    
    a = max((int( (num_vehicles * round) / num_rounds ) + 1), 2)

    return min(a, num_vehicles)


for x in range(1, 11):
    print(vehicles_in_round(10, 100, x))