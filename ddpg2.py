import numpy
import tensorflow as tf
from qmaze import Qmaze, completion_check
from experience import *
from quadtree import leaves
import keras.backend as K
from tensorflow.keras import layers


class DDPG:
    def __init__(self, actor, critic, actor_target, critic_target, quadtree, start, goal, tau, gamma = 0.99):
        self.actor = actor
        self.critic = critic
        self.actor_target = actor_target
        self.critic_target = critic_target
        self.qmaze = Qmaze(quadtree, start, goal)
        self.start = start
        self.goal = goal
        self.tau = tau
        self.gamma = gamma

    def gradients(self, states, actions):
        """ Compute Q-value gradients w.r.t. states and policy-actions
        """

        return self.action_grads([states, actions])

    def transfer_weights(self, model, target_model):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = model.get_weights(), target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau) * target_W[i]
        target_model.set_weights(target_W)

    def bellman(self, rewards, q_values, dones):
        """ Use the Bellman Equation to compute the critic target
        """
        critic_target = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                critic_target[i] = rewards[i]
            else:
                critic_target[i] = rewards[i] + self.gamma * q_values[i]
        return critic_target

    def update_models(self, states, actions, critic_target):
        """ Update actor and critic networks from sampled experience
        """

        # Train critic
        self.critic.train_on_batch([states, actions], critic_target)
        # Q-Value Gradients under Current Policy
        actions = self.actor.predict(states)
        grads = self.gradients(states, actions)
        # Train actor
        self.actor.adam_optimizer([states, np.array(grads).reshape((-1, self.qmaze.size))])
        # Transfer weights to target networks at rate Tau
        self.transfer_weights(self.actor, self.actor_target)
        self.transfer_weights(self.critic, self.critic_target)

    def train(self, **opt):
        max_memory = opt.get('max_memory', 100000)
        n_epochs = opt.get('n_epoch', 10)
        data_size = opt.get('data_size', 1000)
        actor_name = opt.get('actor_name', 'actor_model')
        critic_name = opt.get('critic_name', 'critic_model')
        epsilon = opt.get('epsilon', 0.1)

        self.transfer_weights(self.actor, self.actor_target)
        self.transfer_weights(self.critic, self.critic_target)

        experience = Experience(self.actor_target, max_memory=max_memory)
        win_history = []
        n_episodes = 0
        game_over = False
        hsize = self.qmaze.size // 2
        start_time = datetime.datetime.now()

        for epoch in range(n_epochs):
            envstate = self.qmaze.observe()
            self.qmaze.reset(self.start)

            while not game_over:
                loss = 0.0
                valid_actions = self.qmaze.valid_actions()
                prev_envstate = envstate

                if np.random.rand() < epsilon:
                    # print("Exploring")
                    temp = []
                    for leaf1 in range(len(leaves)):
                        for leaf2 in valid_actions:
                            if leaves[leaf1].center() == leaf2.center():
                                temp.append(leaf1)
                    action = random.choice(temp)
                else:
                    # print("Exploiting")
                    temp = experience.predict(prev_envstate)
                    action = np.argmax(temp)

                envstate, reward, game_status = self.qmaze.act(action)

                if game_status == 'win':
                    win_history.append(1)
                    # print("Win")
                    game_over = True
                elif game_status == 'lose':
                    win_history.append(0)
                    # print("Lose")
                    game_over = True
                else:
                    game_over = False

                episode = [prev_envstate, action, reward, envstate, game_over]
                experience.remember(episode)
                n_episodes += 1

                states, targets, actions, rewards, dones = experience.get_data_ddpg(data_size=data_size)

                q_values = self.critic_target.predict([targets, self.actor_target.predict(targets)])
                critic_target = self.bellman(rewards[0], q_values[0], dones[0])

                self.update_models(states, actions, critic_target)
                loss = self.critic.evaluate(states, targets, verbose=0)

            if len(win_history) > hsize:
                win_rate = sum(win_history[-hsize:]) / hsize

            dt = datetime.datetime.now() - start_time
            t = format_time(dt.total_seconds())
            template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
            print(template.format(epoch, n_epochs - 1, loss, n_episodes, sum(win_history), win_rate, t))
            # we simply check if training has exhausted all free cells and if in all
            # cases the agent won
            if win_rate > 0.9: epsilon = 0.05
            # if sum(win_history[-hsize:]) == hsize and completion_check(model, qmaze):
            if sum(win_history[-hsize:]) == hsize:
                print("Reached 100%% win rate at epoch: %d" % (epoch,))
                break

        h5file = actor_name + ".h5"
        json_file = actor_name + ".json"
        self.actor.save_weights(h5file, overwrite=True)
        with open(json_file, "w") as outfile:
            json.dump(self.actor.to_json(), outfile)

        h5file = critic_name + ".h5"
        json_file = critic_name + ".json"
        self.critic.save_weights(h5file, overwrite=True)
        with open(json_file, "w") as outfile:
            json.dump(self.critic.to_json(), outfile)

        end_time = datetime.datetime.now()
        dt = datetime.datetime.now() - start_time
        seconds = dt.total_seconds()
        t = format_time(seconds)
        print('files: %s, %s' % (h5file, json_file))
        print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (epoch, max_memory, data_size, t))
        return seconds

def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)



