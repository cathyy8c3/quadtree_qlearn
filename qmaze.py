from graph import make_adjacent_function, euclidian
from quadtree import *
from experience import Experience
import random

class Qmaze(object):
    def __init__(self, quadtree, start, goal):
        self._quadtree = quadtree
        self.size = len(leaves)
        self.target = goal  # target cell where the "cheese" is
        self.free_cells = self._quadtree.passable_cells()
        self.determine = False
        self.adjfunc = make_adjacent_function(quadtree)

        if start.center() == goal.center():
            raise Exception("Invalid Location: cannot route to itself")
        if self.target.color != PASSABLE:
            raise Exception("Invalid maze: target cell cannot be blocked!")
        if start.color != PASSABLE:
            raise Exception("Invalid Location: must sit on a free cell")
        if not floodfill(start, goal, quadtree):
            raise Exception("Goal is unreachable")
        self.reset(start)

    def reset(self, state):
        self.quadtree = copy.deepcopy(self._quadtree)
        self.size = len(leaves)
        self.state = (state, 'valid')
        self.min_reward = -0.5 * self.size
        self.total_reward = 0
        self.visited = set()

    def update_state(self, action):
        size = self.size
        curr_state, mode = self.state
        next_state = curr_state
        nmode = mode

        # if self.determine:
        if curr_state.color == PASSABLE:
            self.visited.add(curr_state)  # mark visited cell

        valid_actions = self.valid_actions()

        valid = False
        for leaf in valid_actions:
            if leaf.center() == leaves[action].center():
                valid = True

        if not valid_actions:
            nmode = 'blocked'
        elif valid:
            nmode = 'valid'
            next_state = leaves[action]
        else:  # invalid action, no change in position
            nmode = 'invalid'

        # new state
        self.state = (next_state, nmode)
        # else:

    def get_reward(self):
        curr_state, mode = self.state
        size = self.size

        visited = False
        for leaf in self.visited:
            if curr_state.center() == leaf:
                visited = True

        if curr_state.center() == self.target.center():
            return 1.0
        if mode == 'blocked':
            return self.min_reward - 1
        if visited:
            return -0.25
        if mode == 'invalid':
            return -0.75
        if mode == 'valid':
            return -0.04

    def act(self, action):
        # action = self.next_pos(action)
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()

        # curr_state, mode = self.state
        # print("Current : " + str(curr_state) + "\t" + str(mode))
        # print("Goal: " + str(self.target))
        # print(self.valid_actions())
        # print("Reward: " + str(self.total_reward))
        # print("Status: " + str(status))
        # print(envstate)
        # print()

        return envstate, reward, status

    def act_game(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()

        return envstate, reward, status

    def observe(self):
        curr_state, mode = self.state
        return self.quadtree.flatten(curr_state, self.target)

    def game_status(self):
        if self.total_reward < self.min_reward:
            return 'lose'
        curr_state, mode = self.state
        size = self.size

        if curr_state.center() == self.target.center():
            return 'win'

        return 'not_over'

    def valid_actions(self, cell=None):
        if cell is None:
            curr_state, mode = self.state
        else:
            curr_state = cell

        actions = self.adjfunc(curr_state)

        return actions

    def action_prob(self, heurfunc, action):
        curr_state, mode = self.state
        goal = self.target
        valid_actions = self.valid_actions(curr_state)
        dist = dict()
        total_dist = 0.0
        curr_dist = heurfunc(leaves[action], goal)

        for leaf in valid_actions:
            if heurfunc(leaf, goal) <= curr_dist:
                dist[leaf] = heurfunc(leaf, goal)
                total_dist += dist[leaf]

        keys = []
        vals = []
        for key in dist.keys():
            keys.append(key)
            try:
                d = 1/dist[key]
            except:
                d = 1.0
            vals.append(d)

        # print(keys)
        # print(vals)
        # # # print()
        # print(dist)
        # print(total_dist)
        # # # print(random.choices(keys, weights = vals)[0])
        # # # print(type(random.choices(keys, weights = vals)[0]))
        # print()
        if len(keys) == 0:
            return action
        elif sum(vals) == 0:
            return action
        return random.choices(keys, weights = vals)[0].get_index()

    def next_pos(self, action):
        if self.determine:
            next_state = action
            self.determine = False
        else:
            action2 = self.action_prob(euclidian, action)
            self.determine = True
            next_state = self.next_pos(action2)

        return next_state

def play_game(model, qmaze, start):
    qmaze.reset(start)

    envstate = qmaze.observe()
    while True:
        prev_envstate = envstate
        # get next action
        q = model.predict(prev_envstate)
        action = np.argmax(q[0])

        # apply action, get rewards and new state
        envstate, reward, game_status = qmaze.act_game(action)

        print(envstate)
        print(action)
        print(leaves[action])
        print(qmaze.target)

        # print(game_status)
        if game_status == 'win':
            # print("Win")
            return True
        elif game_status == 'lose':
            # print("Lose")
            return False

def run_game(model, qmaze, start):
    qmaze.reset(start)
    path = []
    path2 = []

    envstate = qmaze.observe()
    experience = Experience(model)
    while True:
        prev_envstate = envstate

        # get next action
        temp = model.predict(prev_envstate)
        q = model.predict(prev_envstate)[0]
        action = np.argmax(q)

        # apply action, get rewards and new state
        envstate, reward, game_status = qmaze.act_game(action)

        curr_state, mode = qmaze.state
        path.append(curr_state)

        print(prev_envstate)
        print(str(action) + "\t" + str(mode))
        print(q)
        print(temp)
        print(leaves)
        print(game_status)
        print()

        if game_status == 'win':
            return path
        elif game_status == 'lose':
            return path

def completion_check(model, qmaze):
    for cell in qmaze.free_cells:
        if not qmaze.valid_actions(cell):
            print("Not valid actions")
            return False
        if not play_game(model, qmaze, cell):
            print("Not play game")
            return False
    return True

def floodfill(start, goal, quadtree):
    visited = set()
    queue = []

    adj = make_adjacent_function(quadtree)

    queue.append(start)
    visited.add(start)

    while queue:
        next_leaf = queue.pop(0)
        # print("Next: " + str(next_leaf))
        visited.add(next_leaf)
        # print(visited)

        for leaf1 in adj(next_leaf):
            if (not leaf1 in visited) and leaf1.color == PASSABLE:
                queue.append(leaf1)

    for leaf2 in visited:
        if leaf2.center() == goal.center():
            return True

    # print(visited)

    return False