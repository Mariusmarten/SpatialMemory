def count_actions(data):
    count = []
    for i in set(data['actions']):
        count.append(data['actions'].count(i))
    return count

def shapes(x, features, hidden, out0):
    print('input cnn:', x.shape, '- Batch size, Channel out, Height out, Width out')
    print('output cnn:', features.shape, " - Batch size, sequence length, input size")
    print('input lstm:', features.shape, " - Batch size, sequence length, input size")
    print('hidden lstm:', hidden[0].shape)
    print('output lstm:', out0.shape, '\n')
