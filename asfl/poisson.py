

def vehicles_in_round(num_rounds, num_vehicles, round):
    
    return (num_vehicles * round) / num_rounds


for i in range(0,10):
    print (vehicles_in_round(10, 100, i))