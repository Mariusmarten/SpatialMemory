from collections import Counter


def count_actions(data):
    count = []
    for i in set(data["actions"]):
        count.append(data["actions"].count(i))
    return count


def shapes(x, features, hidden, out0):
    print("input cnn:", x.shape, "- Batch size, Channel out, Height out, Width out")
    print("output cnn:", features.shape, " - Batch size, sequence length, input size")
    print("input lstm:", features.shape, " - Batch size, sequence length, input size")
    print("hidden lstm:", hidden[0].shape)
    print("output lstm:", out0.shape, "\n")


def determine_overlap_train_test_data(train_data, test_data):

    actions = []
    for element in train_data["actions"]:
        actions.append(str(element[:]))

    d1 = Counter(actions)
    count = 0
    for key, value in d1.items():
        count += value
    print("total train data:", count)

    actions = []
    for element in test_data["actions"]:
        actions.append(str(element[:]))

    d2 = Counter(actions)
    count = 0
    for key, value in d2.items():
        count += value
    print("total test data:", count)

    d = {x: d1[x] for x in d1 if x in d2}
    count = 0
    for key, value in d.items():
        count += value

    return count
