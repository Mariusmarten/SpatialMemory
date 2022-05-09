import torch
import plot
from tqdm.auto import tqdm

import torchvision
from torch.utils.tensorboard import SummaryWriter

def train(train_data, val_data, net, criterion, optimizer, steps):
    '''
    Main training loop
    Input: dataset_loader, network, training_loss, optimizer, step size
    Output: trained network
    '''

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    with tqdm(total=steps, unit =" Episode", desc ="Progress") as pbar:
        for epoch in range(steps):  # loop over the dataset multiple times

            train_running_loss = 0.0

            train_correct = 0
            train_total = 0

            for i, data in enumerate(train_data, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

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
                        inputs, labels = data
                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                        test_running_loss += loss.item()

                        _, predicted = torch.max(outputs.data, 1)
                        test_total += labels.size(0)
                        test_correct += (predicted == labels).sum().item()

            train_loss.append(train_running_loss/len(train_data))
            test_loss.append(test_running_loss/len(val_data))
            train_acc.append(100 * train_correct / train_total)
            test_acc.append(100 * test_correct / test_total)
            pbar.update(1)

            if epoch % 10 == 0:
                print(f'Epoch: {epoch + 1}, Train Loss: {(train_running_loss/len(train_data)):.4}, Train Acc: {(100 * train_correct / train_total):.4} %,  Test Loss: {(test_running_loss/len(val_data)):.4}, Test Acc: {(100 * test_correct / test_total):.4} %,')

    print('Finished Training')
    return net, train_loss, test_loss, train_acc, test_acc


def DualOutput(train_data, val_data, net, criterion, optimizer, steps):
    '''
    Main training loop
    Input: dataset_loader, network, training_loss, optimizer, step size
    Output: trained network
    '''

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    with tqdm(total=steps, unit =" Episode", desc ="Progress") as pbar:
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

                # track loss statistics
                train_running_loss += loss.item()

                # plot gradients in each layer
                plot.plot_grad_flow(net.named_parameters())

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

                        # compute test acc
                        outputs_A = torch.round(outputs_A.data)
                        outputs_B = torch.round(outputs_B.data)

                        labels_A = torch.round(labels_A)
                        labels_B = torch.round(labels_B)

                        test_total += labels_A.size(0) + labels_B.size(0)

                        test_correct += int((outputs_A.squeeze() == labels_A).sum())
                        test_correct += int((outputs_B.squeeze() == labels_B).sum())

            train_loss.append(train_running_loss/len(train_data))
            test_loss.append(test_running_loss/len(val_data))
            train_acc.append(100 * train_correct / train_total)
            test_acc.append(100 * test_correct / test_total)
            pbar.update(1)

            if epoch % 10 == 0:
                print(f'Epoch: {epoch + 1}, Train Loss: {(train_running_loss/len(train_data)):.4}, Train Acc: {(100 * train_correct / train_total):.4} %,  Test Loss: {(test_running_loss/len(val_data)):.4}, Test Acc: {(100 * test_correct / test_total):.4} %,')

    print('Finished Training')
    return net, train_loss, test_loss, train_acc, test_acc


def DualInput(train_data, val_data, net, criterion, optimizer, steps):
    '''
    Main training loop
    Input: dataset_loader, network, training_loss, optimizer, step size
    Output: trained network
    '''

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    writer = SummaryWriter()

    imagesA, imagesB, labels = next(iter(train_data))
    grid = torchvision.utils.make_grid(imagesA)
    writer.add_image('images', grid, 0)
    writer.add_graph(net, [imagesA, imagesB])

    with tqdm(total=steps, unit =" Episode", desc ="Progress") as pbar:
        for epoch in range(steps):  # loop over the dataset multiple times

            train_running_loss = 0.0

            train_correct = 0
            train_total = 0

            for i, data in enumerate(train_data, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputsA, inputB, labels = data
                labels = torch.FloatTensor(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputsA, inputB)

                # loss
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
                        inputs, labels = data
                        outputs = net(inputs, inputs)
                        loss = criterion(outputs, labels)
                        test_running_loss += loss.item()

                        _, predicted = torch.max(outputs.data, 1)
                        test_total += labels.size(0)
                        test_correct += (predicted == labels).sum().item()

            # tensorboard logs
            writer.add_scalar('Loss/train', (train_running_loss/len(train_data)), epoch)
            writer.add_scalar('Loss/test', (test_running_loss/len(val_data)), epoch)
            writer.add_scalar('Accuracy/train', (100 * train_correct / train_total), epoch)
            writer.add_scalar('Accuracy/test', (100 * test_correct / test_total), epoch)

            # plotting logs
            train_loss.append(train_running_loss/len(train_data))
            test_loss.append(test_running_loss/len(val_data))
            train_acc.append(100 * train_correct / train_total)
            test_acc.append(100 * test_correct / test_total)
            pbar.update(1)

            if epoch % 10 == 0:
                print(f'Epoch: {epoch + 1}, Train Loss: {(train_running_loss/len(train_data)):.4}, Train Acc: {100 * train_correct // train_total} %,  Test Loss: {(test_running_loss/len(val_data)):.4}, Test Acc: {100 * test_correct // test_total} %,')

    writer.close()
    print('Finished Training')
    return net, train_loss, test_loss, train_acc, test_acc
