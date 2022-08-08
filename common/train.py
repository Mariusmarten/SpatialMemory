import torch
import common.plot
from tqdm.auto import tqdm

import torchvision
from torch.utils.tensorboard import SummaryWriter


def train_Feedforward(train_data, val_data, net, criterion, optimizer, steps):
    """
    Main training loop
    Input: dataset_loader, network, training_loss, optimizer, step size
    Output: trained network
    """

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    with tqdm(total=steps, unit=" Episode", desc="Progress") as pbar:
        for epoch in range(steps):  # loop over the dataset multiple times

            train_running_loss = 0.0

            train_correct = 0
            train_total = 0

            for i, data in enumerate(train_data, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, _, labels = data
                labels = labels[0]
                labels = labels.to(torch.long)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # compute acc
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                # track loss statistics
                train_running_loss += loss.item()

                test_running_loss = 0.0

                test_correct = 0
                test_total = 0

                # same for validation set
                with torch.no_grad():
                    for data in val_data:
                        inputs, _, labels = data
                        labels = labels[0]
                        labels = labels.to(torch.long)

                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                        test_running_loss += loss.item()

                        _, predicted = torch.max(outputs.data, 1)
                        test_total += labels.size(0)
                        test_correct += (predicted == labels).sum().item()

            train_loss.append(train_running_loss / len(train_data))
            test_loss.append(test_running_loss / len(val_data))
            train_acc.append(100 * train_correct / train_total)
            test_acc.append(100 * test_correct / test_total)
            pbar.update(1)

            if epoch % 10 == 0:
                print(
                    f"Epoch: {epoch + 1}, Train Loss: {(train_running_loss/len(train_data)):.4}, Train Acc: {(100 * train_correct / train_total):.4} %,  Test Loss: {(test_running_loss/len(val_data)):.4}, Test Acc: {(100 * test_correct / test_total):.4} %,"
                )

    print("Finished Training")
    return net, train_loss, test_loss, train_acc, test_acc


def train_DiffImg(train_data, val_data, net, criterion, optimizer, steps):
    """
    Main training loop
    Input: dataset_loader, network, training_loss, optimizer, step size
    Output: trained network
    """

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    with tqdm(total=steps, unit=" Episode", desc="Progress") as pbar:
        for epoch in range(steps):  # loop over the dataset multiple times

            train_running_loss = 0.0

            train_correct = 0
            train_total = 0

            for i, data in enumerate(train_data, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputsA, inputsB, labels = data
                labels = labels[0]
                labels = labels.to(torch.long)

                inputs = inputsB - inputsA

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # compute acc
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                # track loss statistics
                train_running_loss += loss.item()

                test_running_loss = 0.0

                test_correct = 0
                test_total = 0

                # same for validation set
                with torch.no_grad():
                    for data in val_data:
                        inputsA, inputsB, labels = data
                        labels = labels[0]
                        labels = labels.to(torch.long)

                        inputs = inputsB - inputsA

                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                        test_running_loss += loss.item()

                        _, predicted = torch.max(outputs.data, 1)
                        test_total += labels.size(0)
                        test_correct += (predicted == labels).sum().item()

            train_loss.append(train_running_loss / len(train_data))
            test_loss.append(test_running_loss / len(val_data))
            train_acc.append(100 * train_correct / train_total)
            test_acc.append(100 * test_correct / test_total)
            pbar.update(1)

            if epoch % 10 == 0:
                print(
                    f"Epoch: {epoch + 1}, Train Loss: {(train_running_loss/len(train_data)):.4}, Train Acc: {(100 * train_correct / train_total):.4} %,  Test Loss: {(test_running_loss/len(val_data)):.4}, Test Acc: {(100 * test_correct / test_total):.4} %,"
                )

    print("Finished Training")
    return net, train_loss, test_loss, train_acc, test_acc


def train_DualOutput(train_data, val_data, net, criterion, optimizer, steps):
    """
    Main training loop
    Input: dataset_loader, network, training_loss, optimizer, step size
    Output: trained network
    """

    train_loss = []
    test_loss = []

    train_acc = []
    test_acc = []

    train_distances = []
    test_distances = []

    train_distances_itemwise = []
    test_distances_itemwise = []

    with tqdm(total=steps, unit=" Episode", desc="Progress") as pbar:
        for epoch in range(steps):  # loop over the dataset multiple times

            train_running_loss = 0.0

            train_correct = 0
            train_total = 0

            for i, data in enumerate(train_data, 0):
                # get the inputs: data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs_A, outputs_B = net(inputs)

                # get target values (y1, _, y2)
                labels_A = torch.FloatTensor([item[0] for item in labels])
                labels_B = torch.FloatTensor([item[2] for item in labels])

                # compute losses separately
                loss_A = criterion(outputs_A.squeeze(), labels_A)
                loss_B = criterion(outputs_B.squeeze(), labels_B)

                loss = loss_A + loss_B
                loss.backward()

                # compute train_distances
                train_distance_itemwise = torch.sqrt(
                    torch.square(labels_A - outputs_A.squeeze())
                    + torch.square(labels_B - outputs_B.squeeze())
                )
                train_distance = torch.sum(train_distance_itemwise)
                train_distance = torch.div(train_distance, len(labels_A))

                # track loss statistics
                train_running_loss += loss.item()

                # plot gradients in each layer
                common.plot.plot_grad_flow(net.named_parameters())

                optimizer.step()

                # compute train acc
                outputs_A = torch.round(outputs_A.data)
                outputs_B = torch.round(outputs_B.data)

                labels_A = torch.round(labels_A)
                labels_B = torch.round(labels_B)

                train_total += labels_A.size(0) + labels_B.size(0)

                train_correct += int((outputs_A.squeeze() == labels_A).sum())
                train_correct += int((outputs_B.squeeze() == labels_B).sum())

                test_running_loss = 0.0

                test_correct = 0
                test_total = 0

                # same for validation set
                with torch.no_grad():
                    for data in val_data:
                        inputs, labels = data

                        # compute test loss
                        labels_A = torch.FloatTensor([item[0] for item in labels])
                        labels_B = torch.FloatTensor([item[2] for item in labels])

                        outputs_A, outputs_B = net(inputs)

                        loss_A = criterion(outputs_A.squeeze(), labels_A)
                        loss_B = criterion(outputs_B.squeeze(), labels_B)

                        loss = loss_A + loss_B
                        test_running_loss += loss.item()

                        # compute test_distances
                        test_distance_itemwise = torch.sqrt(
                            torch.square(labels_A - outputs_A.squeeze())
                            + torch.square(labels_B - outputs_B.squeeze())
                        )
                        test_distance = torch.sum(test_distance_itemwise)
                        test_distance = torch.div(test_distance, len(labels_A)).numpy()

                        # compute test acc
                        outputs_A = torch.round(outputs_A.data)
                        outputs_B = torch.round(outputs_B.data)

                        labels_A = torch.round(labels_A)
                        labels_B = torch.round(labels_B)

                        test_total += labels_A.size(0) + labels_B.size(0)

                        test_correct += int((outputs_A.squeeze() == labels_A).sum())
                        test_correct += int((outputs_B.squeeze() == labels_B).sum())

            train_loss.append(train_running_loss / len(train_data))
            test_loss.append(test_running_loss / len(val_data))

            train_acc.append(100 * train_correct / train_total)
            test_acc.append(100 * test_correct / test_total)

            train_distances.append(train_distance)
            test_distances.append(test_distance)

            train_distances_itemwise.append(train_distance_itemwise)
            test_distances_itemwise.append(test_distance_itemwise)

            pbar.update(1)

            if epoch % 10 == 0:
                print(
                    f"Epoch: {epoch + 1}, Train Loss: {(train_running_loss/len(train_data)):.4}, Train Acc: {(100 * train_correct / train_total):.4} %,  Test Loss: {(test_running_loss/len(val_data)):.4}, Test Acc: {(100 * test_correct / test_total):.4} %,"
                )

    print("Finished Training")
    return (
        net,
        train_loss,
        test_loss,
        train_acc,
        test_acc,
        train_distances,
        test_distances,
        train_distances_itemwise,
        test_distances_itemwise,
    )


def train_DualInput(train_data, val_data, net, criterion, optimizer, steps):
    """
    Main training loop
    Input: dataset_loader, network, training_loss, optimizer, step size
    Output: trained network
    """

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    writer = SummaryWriter()

    imagesA, imagesB, labels = next(iter(train_data))
    grid = torchvision.utils.make_grid(imagesA)
    writer.add_image("images", grid, 0)
    writer.add_graph(net, [imagesA, imagesB])

    with tqdm(total=steps, unit=" Episode", desc="Progress") as pbar:
        for epoch in range(steps):  # loop over the dataset multiple times

            train_running_loss = 0.0

            train_correct = 0
            train_total = 0

            for i, data in enumerate(train_data, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputsA, inputB, labels = data
                labels = labels[0]
                labels = labels.to(torch.long)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputsA, inputB)

                # loss
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()

                # compute acc
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                # track loss statistics
                train_running_loss += loss.item()

                test_running_loss = 0.0

                test_correct = 0
                test_total = 0

                # same for validation set
                with torch.no_grad():
                    for data in val_data:
                        inputsA, inputsB, labels = data
                        labels = labels[0]
                        labels = labels.to(torch.long)

                        outputs = net(inputsA, inputsB)
                        loss = criterion(outputs.squeeze(), labels)
                        test_running_loss += loss.item()

                        _, predicted = torch.max(outputs.data, 1)
                        test_total += labels.size(0)
                        test_correct += (predicted == labels).sum().item()

            # tensorboard logs
            writer.add_scalar(
                "Loss/train", (train_running_loss / len(train_data)), epoch
            )
            writer.add_scalar("Loss/test", (test_running_loss / len(val_data)), epoch)
            writer.add_scalar(
                "Accuracy/train", (100 * train_correct / train_total), epoch
            )
            writer.add_scalar("Accuracy/test", (100 * test_correct / test_total), epoch)

            # plotting logs
            train_loss.append(train_running_loss / len(train_data))
            test_loss.append(test_running_loss / len(val_data))
            train_acc.append(100 * train_correct / train_total)
            test_acc.append(100 * test_correct / test_total)
            pbar.update(1)

            if epoch % 10 == 0:
                print(
                    f"Epoch: {epoch + 1}, Train Loss: {(train_running_loss/len(train_data)):.4}, Train Acc: {(100 * train_correct / train_total):.4} %,  Test Loss: {(test_running_loss/len(val_data)):.4}, Test Acc: {(100 * test_correct / test_total):.4} %,"
                )

    writer.close()
    print("Finished Training")
    return net, train_loss, test_loss, train_acc, test_acc


def train_ConvLSTM(
    train_data,
    val_data,
    net_cnn,
    net_lstm,
    criterion,
    optimizer,
    steps,
    length_trajectory,
):
    """
    Main training loop
    Input: dataset_loader, network, training_loss, optimizer, step size
    Output: trained network
    """

    train_loss = []
    test_loss = []

    train_acc = []
    test_acc = []

    train_distances = []
    test_distances = []

    train_distances_itemwise = []
    test_distances_itemwise = []

    h = torch.randn(2, length_trajectory, 100)
    c = torch.randn(2, length_trajectory, 100)

    with tqdm(total=steps, unit=" Episode", desc="Progress") as pbar:
        for epoch in range(steps):  # loop over the dataset multiple times

            train_running_loss = 0.0

            for i, data in enumerate(train_data, 0):

                # get the inputs; data is a list of [inputs, labels]
                inputs, label = next(iter(train_data))

                # https://discuss.pytorch.org/t/how-to-turn-a-list-of-tensor-to-tensor/8868/4
                labels = torch.stack(label)
                labels = torch.swapaxes(labels, 0, 1)

                # get target values (y1, _, y2) from each of the 64*10 items
                labels_A, labels_B = [], []
                for items in labels:
                    label_A, label_B = [], []
                    for item in items:
                        label_A.append(item[0])
                        label_B.append(item[2])
                    label_A = torch.stack(label_A)
                    label_B = torch.stack(label_B)
                    labels_A.append(label_A)
                    labels_B.append(label_B)

                labels_A = torch.stack(labels_A)
                labels_B = torch.stack(labels_B)
                inputs = torch.stack(inputs)
                inputs = torch.swapaxes(inputs, 0, 1)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                # opt = torch.rand((64, 10, 3, 32, 32))
                # print('shape of the optimal inputs', opt.shape)
                # print('shape of the network inputs', inputs.shape)
                encoded = net_cnn(inputs)
                outputs_A, outputs_B, h, c = net_lstm(encoded, h, c)
                outputs_A = outputs_A.double()
                outputs_B = outputs_B.double()

                # compute losses separatel
                loss_A = criterion(outputs_A.squeeze(), labels_A)
                loss_B = criterion(outputs_B.squeeze(), labels_B)

                loss = loss_A + loss_B
                loss.backward()
                optimizer.step()

                # compute train_distances
                train_distance_itemwise = torch.sqrt(
                    torch.square(labels_A - outputs_A.squeeze())
                    + torch.square(labels_B - outputs_B.squeeze())
                )
                train_distance = torch.sum(train_distance_itemwise)
                train_distance = torch.div(
                    train_distance, len(labels_A) * length_trajectory
                )

                # track loss statistics
                train_running_loss += loss.item()

                test_running_loss = 0.0
                test_correct = 0
                test_total = 0

                # same for validation set
                with torch.no_grad():
                    for data in val_data:

                        # get the inputs; data is a list of [inputs, labels]
                        inputs, label = next(iter(val_data))

                        # https://discuss.pytorch.org/t/how-to-turn-a-list-of-tensor-to-tensor/8868/4
                        labels = torch.stack(label)
                        labels = torch.swapaxes(labels, 0, 1)

                        # get target values (y1, _, y2) from each of the 10
                        labels_A, labels_B = [], []
                        for items in labels:
                            label_A, label_B = [], []
                            for item in items:
                                label_A.append(item[0])
                                label_B.append(item[2])
                            label_A = torch.stack(label_A)
                            label_B = torch.stack(label_B)
                            labels_A.append(label_A)
                            labels_B.append(label_B)
                        labels_A = torch.stack(labels_A)
                        labels_B = torch.stack(labels_B)

                        inputs = torch.stack(inputs)
                        inputs = torch.swapaxes(inputs, 0, 1)

                        encoded = net_cnn(inputs)
                        print(encoded)
                        outputs_A, outputs_B, h, c = net_lstm(encoded, h, c)
                        outputs_A = outputs_A.double()
                        outputs_B = outputs_B.double()

                        loss_A = criterion(outputs_A.squeeze(), labels_A)
                        loss_B = criterion(outputs_B.squeeze(), labels_B)

                        loss = loss_A + loss_B
                        test_running_loss += loss.item()

                        # compute test_distances
                        test_distance_itemwise = torch.sqrt(
                            torch.square(labels_A - outputs_A.squeeze())
                            + torch.square(labels_B - outputs_B.squeeze())
                        )
                        test_distance = torch.sum(test_distance_itemwise)
                        test_distance = torch.div(
                            test_distance, len(labels_A) * length_trajectory
                        ).numpy()

            train_loss.append(train_running_loss / len(train_data) * 10)
            test_loss.append(test_running_loss / len(val_data) * 10)

            train_distances.append(train_distance)
            test_distances.append(test_distance)

            train_distances_itemwise.append(train_distance_itemwise)
            test_distances_itemwise.append(test_distance_itemwise)

            pbar.update(1)

            if epoch % 10 == 0:
                print(
                    f"Epoch: {epoch + 1}, Train Loss: {(train_running_loss/len(train_data)):.4}, Train distance: {train_distance:.4}, , Test distance: {test_distance:.4}"
                )
    print("Finished Training")

    return (
        train_loss,
        test_loss,
        train_distances,
        test_distances,
        train_distances_itemwise,
        test_distances_itemwise,
    )
