import random
import turtle
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

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
