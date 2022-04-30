import copy

def oracle_actions():
    # optimal action seq. for this maze only (seed 0)
    # 0 - turn left, 1 - turn right, 2 - walk forwards, 3 - walk backwards
    actions = []
    actions = actions + [1]*7 # angle 270, current-pos: (2.16, 0.00, 7.49) - turn right
    actions = actions + [2]*70 # angle 270, current-pos: (2.01, 0.00, 16.48) - walk forwards
    actions = actions + [0]*6 # angle 83, current-pos: (1.98, 0.00, 17.98) - turn left
    actions = actions + [2]*20 # angle 0,  current-pos: (4.98, 0.00, 18.04) - walk forwards
    actions = actions + [1]*6 # angle 270, current-pos: (4.98, 0.00, 18.04) - turn right
    actions = actions + [2]*40 # angle 270, current-pos: (4.88, 0.00, 24.03) - walk forwards
    actions = actions + [0]*6 # angle 0, current-pos: (4.88, 0.00, 24.03) - turn left
    actions = actions + [2]*20 # angle 0, current-pos: (7.88, 0.00, 24.09) - walk forwards
    actions = actions + [0]*6 # angle 89, current-pos: (7.88, 0.00, 24.09) - turn left
    actions = actions + [2]*20 # angle 89, current-pos: (7.93, 0.00, 21.09) - walk forwards
    actions = actions + [1]*6 # angle 0, current-pos: (7.93, 0.00, 21.09)- turn right
    actions = actions + [2]*20 # - walk forwards
    actions = actions + [0]*6 #  - turn left
    actions = actions + [2]*20 # - walk forwards
    actions = actions + [0]*6 #  - turn left
    actions = actions + [2]*20 # - walk forwards
    actions = actions + [1]*6 # - turn right
    actions = actions + [2]*50 # - walk forwards
    actions = actions + [1]*6 # - turn right
    actions = actions + [2]*20 # - walk forwards
    actions = actions + [0]*6 #  - turn left
    actions = actions + [2]*70 # - walk forwards
    actions = actions + [1]*6 # - turn right
    actions = actions + [2]*20 # - walk forwards
    actions = actions + [1]*6 # - turn right
    actions = actions + [2]*30 # - walk forwards
    actions = actions + [0]*6 #  - turn left
    actions = actions + [2]*20 # - walk forwards
    actions = actions + [0]*6 #  - turn left
    actions = actions + [2]*20 # - walk forwards
    actions = actions + [1]*6 # - turn right
    actions = actions + [2]*25 # - walk forwards
    actions = actions + [1]*6 # - turn right
    actions = actions + [2]*20 # - walk forwards
    actions = actions + [0]*6 #  - turn left
    actions = actions + [2]*20 # - walk forwards
    actions = actions + [1]*6 # - turn right
    actions = actions + [2]*20 # - walk forwards
    actions = actions + [1]*6 # - turn right
    actions = actions + [2]*20 # - walk forwards
    actions = actions + [0]*6 #  - turn left
    actions = actions + [2]*20 # - walk forwards
    actions = actions + [1]*6 # - turn right
    actions = actions + [2]*20 # - walk forwards
    actions = actions + [0]*6 #  - turn left
    actions = actions + [2]*45 # - walk forwards
    actions = actions + [0]*6 # - turn right
    actions = actions + [2]*20 # - walk forwards
    actions = actions + [0]*6 #  - turn left
    actions = actions + [2]*20 # - walk forwards
    actions = actions + [1]*6 # - turn right
    actions = actions + [2]*20 # - walk forwards
    actions = actions + [1]*6 # - turn right
    actions = actions + [2]*60 # - walk forwards
    actions = actions + [1]*6 # - turn right
    actions = actions + [2]*20 # - walk forwards
    actions = actions + [1]*6 # - turn right
    actions = actions + [2]*20 # - walk forwards
    actions = actions + [0]*6 #  - turn left
    actions = actions + [2]*20 # - walk forwards
    actions = actions + [0]*6 #  - turn left
    actions = actions + [2]*20 # - walk forwards
    actions = actions + [1]*6 # - turn right
    actions = actions + [2]*20 # - walk forwards
    actions = actions + [1]*6 # - turn right
    actions = actions + [2]*100 # - walk forwards
    actions = actions + [1]*6 # - turn right
    actions = actions + [2]*20 # - walk forwards

    return actions

def random_actions(length, env):

    actions = []

    for i in range(length):
        action = env.action_space.sample()
        actions.append(action)

    return actions

def add_randomness(oracle_actions, env, random_step_lenth=3):
    oracle_random_actions = []

    for i in oracle_actions:
        oracle_random_actions.append(i)
        sub_seq = []
        for k in range(random_step_lenth):
            action = env.action_space.sample()
            sub_seq.append(action)

        oracle_random_actions.extend(sub_seq)
        sub_seq = reverse_action_seq(sub_seq)
        oracle_random_actions.extend(sub_seq)

    return oracle_random_actions

def reverse_action_seq(actions):
    act_seq = copy.deepcopy(actions)
    for i in range(len(act_seq)):
        if act_seq[i] == 0:
            act_seq[i] = 1
        elif act_seq[i] == 1:
            act_seq[i] = 0
        elif act_seq[i] == 2:
            act_seq[i] = 3
        elif act_seq[i] == 3:
            act_seq[i] = 2
    act_seq = act_seq[::-1]

    return act_seq
