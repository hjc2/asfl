
cid_ll = [
    (1, ['1', '2', '3', '4', '5', '6', '7']),
    (2, ['1', '2', '3', '4', '5', '6']),
    (3, ['1','2','3']),
    (4, ['4', '5', '6']),
    (5, ['7'])
]
cid_ll = [
    (1, ['1', '2', '3', '4', '5', '6', '7']),
]


def track_node_appearances(data):
    last_appearance = {}
    appearance_count = {}
    current_round = 0

    for round_id, node_list in data[:-1]:
        current_round = round_id
        for node_id in node_list:
            if node_id in last_appearance:
                appearance_count[node_id] = appearance_count.get(node_id, 0) + 1
            last_appearance[node_id] = current_round  # Store the previous round

    return last_appearance


print(track_node_appearances(cid_ll))
