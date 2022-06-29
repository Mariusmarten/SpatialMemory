import math

from tqdm.auto import tqdm
from skimage.transform import resize


def collect(actions, env, img_size=128, show=False):
    """
    Collect dataset containing actions, positions, angles, observations (first-
    person view), depth-view and the map.
    """
    act_col = actions
    pos_col = []
    angl_col = []
    obs_col = []
    top_col = []
    dep_col = []
    length_actions = len(actions)

    my_list = list(range(20))

    with tqdm(total=len(actions), unit=" Steps", desc="Progress") as pbar:
        for i, action in enumerate(actions):
            _, _, _, _ = env.step(action)
            position = env.agent.pos  # coordinates
            angle = int(env.agent.dir * 180 / math.pi) % 360

            observation = env.render("rgb_array")
            top_view = env.render("rgb_array", view="top")  # map
            depth = env.render_depth()

            observation = resize(observation, (img_size, img_size))
            top_view = resize(top_view, (img_size, img_size))
            depth = resize(depth, (img_size, img_size))

            pos_col.append(position)
            angl_col.append(angle)
            obs_col.append(observation)
            top_col.append(top_view)
            dep_col.append(depth)

            pbar.update(1)

            if show and i > length_actions - 7:  # print last n
                plot_obs_top_dep(env)
                print_env_parameters(env)

    dic = {
        "actions": act_col,
        "positions": pos_col,
        "angles": angl_col,
        "observations": obs_col,
        "top_views": top_col,
        "depth_imgs": dep_col,
    }

    return dic
