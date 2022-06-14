def count_actions(data):
    count = []
    for i in set(data['actions']):
        count.append(data['actions'].count(i))
    return count
