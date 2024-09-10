

def vehicles_in_round(num_rounds, num_vehicles, round):
    
    return max(num_vehicles, (int( (num_vehicles * round) / num_rounds ) + 1), 2)

for i in range(1,6):
    print (vehicles_in_round(5, 5, i))