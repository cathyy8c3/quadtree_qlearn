from __init__ import *
from mapgen import PASSABLE, IMPASSABLE
from graph import make_adjacent_function
import copy
from quadtree import *
import quadtree as qtree

class Qmaze(object):
    def __init__(self, quadtree, start, goal):
        self._quadtree = quadtree
        self.size = len(qtree.leaves)
        self.target = goal  # target cell where the "cheese" is
        self.free_cells = self._quadtree.passable_cells()
        if self.target.color != PASSABLE:
            raise Exception("Invalid maze: target cell cannot be blocked!")
        if start.color != PASSABLE:
            raise Exception("Invalid Rat Location: must sit on a free cell")
        self.reset(start)

    def reset(self, state):
        self.state = state
        self.quadtree = copy.deepcopy(self._quadtree)
        self.size = len(qtree.leaves)
        self.state = (state, 'start')
        self.min_reward = -0.5 * self.size
        self.total_reward = 0
        self.visited = set()

    def update_state(self, action):
        size = self.size
        curr_state, mode = self.state
        next_state = curr_state
        nmode = mode

        if curr_state.color == PASSABLE:
            self.visited.add(curr_state)  # mark visited cell

        valid_actions = self.valid_actions()
        # print("Type: " + str(type(valid_actions)))

        # print("Action: " + str(action))
        # print(valid_actions)

        if not valid_actions:
            nmode = 'blocked'
        elif action >= len(valid_actions):
            mode = 'invalid'
        elif valid_actions[action] in valid_actions:
            nmode = 'valid'
            next_state = valid_actions[action]
        else:  # invalid action, no change in rat position
            mode = 'invalid'

        # new state
        self.state = (next_state, nmode)

    def get_reward(self):
        curr_state, mode = self.state
        size = self.size
        if curr_state == self.target:
            return 1.0
        if mode == 'blocked':
            return self.min_reward - 1
        if curr_state in self.visited:
            return -0.25
        if mode == 'invalid':
            return -0.75
        if mode == 'valid':
            return -0.04

    def act(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()

        # print("Reward: " + str(self.total_reward))
        # print("Status: " + str(status))
        return envstate, reward, status

    def observe(self):
        curr_state, mode = self.state
        return self.quadtree.flatten(curr_state)

    # def draw_env(self):
    #     canvas = np.copy(self.quadtree)
    #     nrows, ncols = self.quadtree.shape
    #     # clear all visual marks
    #     for r in range(nrows):
    #         for c in range(ncols):
    #             if canvas[r, c] > 0.0:
    #                 canvas[r, c] = 1.0
    #     # draw the rat
    #     row, col, valid = self.state
    #     canvas[row, col] = rat_mark
    #     return canvas

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

        temp = make_adjacent_function(self.quadtree)
        actions = temp(curr_state)

        return actions

# def show(qmaze):
#     plt.grid('on')
#     nrows, ncols = qmaze.quadtree.shape
#     ax = plt.gca()
#     ax.set_xticks(np.arange(0.5, nrows, 1))
#     ax.set_yticks(np.arange(0.5, ncols, 1))
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     canvas = np.copy(qmaze.quadtree)
#     for row,col in qmaze.visited:
#         canvas[row,col] = 0.6
#     rat_row, rat_col, _ = qmaze.state
#     canvas[rat_row, rat_col] = 0.3   # rat cell
#     canvas[nrows-1, ncols-1] = 0.9 # cheese cell
#     img = plt.imshow(canvas, interpolation='none', cmap='gray')
#     print("Plot shown")
#     return img

def play_game(model, qmaze, start):
    qmaze.reset(start)
    path = []

    envstate = qmaze.observe()
    while True:
        prev_envstate = envstate
        # get next action
        q = model.predict(prev_envstate)
        action = np.argmax(q[0])

        valid_actions = qmaze.valid_actions()
        move = valid_actions[action]
        path.append(move)

        # apply action, get rewards and new state
        envstate, reward, game_status = qmaze.act(action)
        if game_status == 'win':
            return path
        elif game_status == 'lose':
            return False

def completion_check(model, qmaze):
    for cell in qmaze.free_cells:
        if not qmaze.valid_actions(cell):
            return False
        if not play_game(model, qmaze, cell):
            return False
    return True