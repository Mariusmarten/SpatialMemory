import random
import turtle
import torch
import statistics
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter
from PIL import Image

def plot_obs_top_dep(env):

    obs = env.render('rgb_array')
    top = env.render('rgb_array', view='top')
    dep = env.render_depth()

    # plotting
    plt.rcParams['figure.figsize'] = 20, 5.5
    plt.rcParams.update({'font.size': 15})

    plt.figure() #figsize=(20, 5.5)
    fig, axs = plt.subplots(1, 3)
    fig.suptitle('Gym-MiniWorld Environment', fontsize=20)

    axs[0].imshow(obs)
    axs[0].set_title('First-person view (observations)')
    axs[1].imshow(dep)
    axs[1].set_title('First-person depth view')
    axs[2].imshow(top)
    axs[2].set_title('Birds-eye view (map)')

    plt.show()

def plot_3x3_examples(dic):
    '''
    Plots 3 views from each of the 3 images available in each state,
    namely first-person, depth, and the map.
    '''
    fig = plt.figure(figsize=(16, 13))
    columns = 3
    rows = 3
    ax = []
    for i in range(1, columns*rows +1):
        ax.append(fig.add_subplot(rows, columns, i))

        if i in [1, 4, 7]:
            rand = random.randint(0, len(dic['actions'])-1)    # generate random number to pick image
            img = dic['observations'][rand]
            string = 'First-person view. '
        elif i in [3, 6, 9]:
            img = dic['top_views'][rand]
            string = 'Birds-eye view (map). '
        elif i in [2, 5, 8]:
            img = dic['depth_imgs'][rand]
            string = 'First-person depth view. '

        #ax[i].set_title('First-person view (observations)')
        ax[-1].set_title(string+"Step: "+str(rand))
        ax[-1].set_axis_off()
        fig.suptitle('Gym-MiniWorld Environment', fontsize=20)
        plt.imshow(img)
    plt.show()

def plot_64_observations(images):
    '''
    Can be used to plot a full batch. Currently fixed to 64.
    '''
    plt.figure(figsize=(10,10)) # specifying the overall grid size

    for i in range(64):
        plt.subplot(8,8,i+1)    # the number of images in the grid is 64
        plt.imshow(images[i].permute(2, 0, 1).permute(2, 0, 1))
        plt.axis('off')
    plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)
    plt.show()

def save_gif_of_sequence(actions, env_string):
    '''
    Takes as input an action sequence and the name of the environments.
    The functions saves the gifs in the current location.
    '''
    imgs_obs = []
    imgs_top = []
    imgs_depth = []

    env = init_env(env_string)
    for action in actions:
        _, _, _, _ = env.step(action)
        obs = env.render('rgb_array')
        top = env.render('rgb_array', view='top')
        depth = env.render_depth()

        obs = resize(obs, (85, 85))
        top = resize(top, (85, 85))
        depth = resize(depth, (85, 85))

        imgs_obs.append(obs)
        imgs_top.append(top)
        imgs_depth.append(depth)

    imageio.mimsave('agent_view_oracle.gif', imgs_obs, fps=100)
    imageio.mimsave('map_view_oracle.gif', imgs_top, fps=100)
    imageio.mimsave('depth_view_oracle.gif', imgs_depth, fps=100)

def turtle_tracing(oracle_actions):

    oracle_actions = [0] + oracle_actions

    turtle.bgpic("maze-environment.png")

    t1 = turtle.Turtle()
    t1.pensize(5)
    t1.penup()
    t1.setx(-350)
    t1.sety(175)
    t1.pendown()
    t1.color('black')

    dic = {0: '.left(15)', 1: '.right(15)', 2: '.forward(4.75)', 3: '.forward(-4.75)'}
    for action in oracle_actions:
            exec('t1' + dic[action])

    turtle.done()

    try:
        turtle.bye()
    except:
        print("bye")

def plot_losses(test_loss, train_loss):
    with plt.style.context('ggplot'):
        plt.figure(figsize=(12, 7))
        #plt.style.use('ggplot')
        plt.rcParams.update({'font.size': 16})
        plt.plot(test_loss, color='slategray', linewidth=2)
        plt.plot(train_loss, color='red', linewidth=2)
        plt.legend(['Training Loss', 'Validation Loss'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.show()

def plot_acc(test_acc, train_acc, smooth=False):

    if smooth:
        test_acc_smoothed = savgol_filter(test_acc, 30, 12)  # window size, polynomial order
        train_acc_smoothed = savgol_filter(train_acc, 30, 12)

    with plt.style.context('ggplot'):
        plt.figure(figsize=(12, 7))
        #plt.style.use('ggplot')
        plt.rcParams.update({'font.size': 16})
        if smooth:
            plt.plot(test_acc_smoothed, color='slategray', linewidth=2)
            plt.plot(train_acc_smoothed, color='red', linewidth=2)
        else:
            plt.plot(test_acc, color='slategray', linewidth=2)
            plt.plot(train_acc, color='red', linewidth=2)
        plt.legend(['Training Acc', 'Validation Acc'])
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy in %')
        plt.title('Accuracy')
        plt.show()

def show_example_classificataions(dataset, net, amount=5):
    classes_expl = {0: 'turn left', 1: 'turn right', 2: 'walk forwards', 3: 'walk backwards'}

    # calculate total accuracy on training data
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataset:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('GroundTruth: ', ', '.join(f'{classes_expl[int(labels[j])]}' for j in range(amount)))
    print('Predicted: ', ', '.join(f'{classes_expl[int(predicted[j])]}' for j in range(amount)), '\n')

def plot_confusion_matrix(dataset, network, save=False):
    '''
    Plots a confusion matrix.
    Input: dataset (train or test), the trained network
    '''
    classes_expl = {0: 'turn left', 1: 'turn right', 2: 'walk forwards', 3: 'walk backwards'}

    # plot confusion matrix
    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in dataset:
            output = network(inputs) # Feed pass through network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth

    # Build confusion matrix
    classes = list(set(labels))
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [classes_expl[i] for i in classes],
                         columns = [classes_expl[i] for i in classes])
    plt.figure(figsize = (12,7))
    plt.rcParams.update({'font.size': 16})
    sn.heatmap(df_cm, annot=True)
    plt.suptitle('Confusion matrix')
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.show()

    if save:
        plt.savefig('output.png')

def plot_grad_flow(named_parameters):
    '''
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    '''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

def plot_grayscale_image(env):
    image = env.render('rgb_array', view='top')
    arr = np.asarray(image)
    plt.imshow(arr[:,:,2], cmap='gray')
    plt.show()

def plot_euclidean_distance(train_dis, test_dis):

    # preprocess values

    train_dis_notensor  = []
    for i in train_dis:
        i = i.detach().numpy()
        i = i.item()
        train_dis_notensor.append(i)

    test_dis_notensor  = []
    for i in test_dis:
        test_dis_notensor.append(float(i))


    plt.rcParams['figure.figsize'] = 20, 5.5

    with plt.style.context('ggplot'):
        plt.plot(train_dis_notensor[10:], label = "Training distance", linewidth=3, color='tab:blue')
        plt.plot(test_dis_notensor[10:], label = "Validation distance", linewidth=3)
        plt.title('Euclidean distance per epoch', fontsize=22)
        plt.legend(fontsize=18)
        plt.show()

def histo_distribution_shift(test_dis_item):

    max_lim = 17

    def has_len(obj):
        return hasattr(obj, '__len__')

    def is_float_try(inp):
        return isinstance(inp, float) or not has_len(input)

    dis_item_notensor  = []
    for item in test_dis_item:
        if len(list(test_dis_item[0].size())) == 2:
            for i in item:
                i = i.detach().numpy()
                if is_float_try(i):
                    dis_item_notensor.extend(i)
                else:
                    dis_item_notensor.append(i)
        else:
            item = item.detach().numpy()
            dis_item_notensor.extend(item)

    length = len(dis_item_notensor)

    plt.rcParams['figure.figsize'] = 20, 5.5
    plt.rcParams.update({'font.size': 25})
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)

    with plt.style.context('ggplot'):
        plt.hist(dis_item_notensor[0:int(length*0.1)], bins=100, rwidth=0.9, color='indianred', range=(0, max_lim), alpha=0.8)
        plt.title('Distance Distribution - First 10%', fontsize=22)
        plt.xlabel('Distance', fontsize=20)
        plt.ylabel('Amount', fontsize=20)
        mean_val = statistics.mean(dis_item_notensor[10:int(length*0.1)])
        plt.text(mean_val*1.1, 2, 'Mean: {:.2f}'.format(mean_val), fontsize=22)
        plt.axvline(mean_val, color='k', linestyle='dashed', linewidth=2)
        plt.show()

    with plt.style.context('ggplot'):
        plt.hist(dis_item_notensor[int(length*0.9):length-1], bins=100, rwidth=0.9, color='indianred', range=(0, max_lim), alpha=0.8)
        plt.title('Distance Distribution - Last 10%', fontsize=22)
        plt.xlabel('Distance', fontsize=20)
        plt.ylabel('Amount', fontsize=20)
        mean_val = statistics.mean(dis_item_notensor[int(length*0.9):length-1])
        plt.text(mean_val*1.1, 2, 'Mean: {:.2f}'.format(mean_val), fontsize=22)
        plt.axvline(mean_val, color='k', linestyle='dashed', linewidth=2)
        plt.show()

def histo_train_val(test_dis_item, train_dis_item):

    max_lim = 17

    def has_len(obj):
        return hasattr(obj, '__len__')

    def declutter(dis_item):
        dis_item_notensor  = []
        for item in dis_item:
            if len(list(dis_item[0].size())) == 2:
                for i in item:
                    i = i.detach().numpy()
                    dis_item_notensor.extend(i)
            else:
                item = item.detach().numpy()
                dis_item_notensor.extend(item)
        length = len(dis_item_notensor)
        return dis_item_notensor

    test_dis_item_notensor = declutter(test_dis_item)
    train_dis_item_notensor = declutter(train_dis_item)

    plt.rcParams['figure.figsize'] = 20, 5.5
    plt.rcParams.update({'font.size': 25})
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)

    with plt.style.context('ggplot'):
        max_lim = 17
        plt.hist(train_dis_item_notensor[10:], bins=50, rwidth=0.9, color='#607c8e', range=(0, max_lim), alpha=0.9)
        plt.xlabel('Distance', fontsize=20)
        plt.ylabel('Amount', fontsize=20)
        plt.title('Training set - Distances (50 bins)', fontsize=22)
        mean_val = statistics.mean(train_dis_item_notensor[10:])
        plt.text(mean_val*1.1, max_lim*0.9, 'Mean: {:.2f}'.format(mean_val), fontsize=22)
        plt.axvline(mean_val, color='k', linestyle='dashed', linewidth=2)
        plt.show()

    with plt.style.context('ggplot'):
        plt.hist(test_dis_item_notensor[10:], bins=50, rwidth=0.9, color='indianred', range=(0, 17), alpha=0.9)
        plt.xlabel('Distance', fontsize=20)
        plt.ylabel('Amount', fontsize=20)
        plt.title('Validation set - Distances (50 bins)', fontsize=22)
        mean_val = statistics.mean(test_dis_item_notensor[10:])
        plt.text(mean_val*1.1, max_lim*0.9, 'Mean: {:.2f}'.format(mean_val), fontsize=22)
        plt.axvline(mean_val, color='k', linestyle='dashed', linewidth=2)
        plt.show()

def plot_difference_img(data, distance):
    rand = random.randint(0, len(data['actions'])-distance-1)

    act = []
    for i in range(distance):
        act.append(data['actions'][rand+i])

    print('actions ', act)

    new = data['observations'][rand] - data['observations'][rand+distance]

    coord = data['positions'][rand+distance]
    x, _, y = coord
    print('position ', [round(x, 3), round(y, 3)])

    plt.figure(figsize=(7, 7))
    plt.imshow(new)
    plt.show()
