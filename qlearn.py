from ddpg2 import DDPG
from qtrain import build_model
from quadtree import leaves
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Reshape, LSTM, Lambda, BatchNormalization, GaussianNoise, Flatten, concatenate

def run_qlearn(quadtree, start, goal):
    use_file = input("Do you want to load weights from an existing file? (y/n)\t")
    if use_file == "y":
        trained = True
    else:
        trained = False

    train = True

    ddpg = build_ddpg(quadtree, start, goal)

    if not trained:
        ddpg.train(n_epoch=100, max_memory=8 * quadtree.count(), data_size=32)
    #     count = 0
    #     for leaf in leaves:
    #         print("Leaf: " + str(count))
    #         count += 1
    #         if not leaf.color == PASSABLE:
    #             continue
    #         elif leaf.center() == start.center():
    #             continue
    #         elif not floodfill(start, leaf, quadtree):
    #             continue
    #
    #         if train:
    #             qtrain(model, quadtree, start, leaf, n_epoch=100, max_memory=8 * quadtree.count(), data_size=32)
    #             train = False
    #             continue
    #         model.load_weights("model.h5")
    #         qtrain(model, quadtree, start, leaf, n_epoch=100, max_memory=8 * quadtree.count(), data_size=32, weights_file="model.h5")
    # else:
    #     model.load_weights("model.h5")

    return ddpg.actor

def build_ddpg(quadtree, start, goal, lr = 0.00005):
    actor = actor_network(len(leaves), len(leaves), len(leaves))
    critic = critic_network(len(leaves), len(leaves))
    actor_target = actor_network(len(leaves), len(leaves), len(leaves))
    critic_target = critic_network(len(leaves), len(leaves))

    actor.compile(Adam(lr), 'mse')
    actor_target.compile(Adam(lr), 'mse')
    critic.compile(Adam(lr), 'mse')
    critic_target.compile(Adam(lr), 'mse')

    tau = 0.001

    return DDPG(actor, critic, actor_target, critic_target, quadtree, start, goal, tau)

def actor_network(env_dim, act_dim, act_range):
    """ Actor Network for Policy function Approximation, using a tanh
    activation for continuous control. We add parameter noise to encourage
    exploration, and balance it with Layer Normalization.
    """
    inp = Input((env_dim))
    #
    x = Dense(256, activation='relu')(inp)
    x = GaussianNoise(1.0)(x)
    #
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = GaussianNoise(1.0)(x)
    #
    out = Dense(act_dim, activation='tanh', kernel_initializer='random_uniform')(x)
    out = Lambda(lambda i: i * act_range)(out)
    #
    return Model(inp, out)

def critic_network(env_dim, act_dim):
    """ Assemble Critic network to predict q-values
    """
    state = Input((env_dim))
    action = Input((act_dim,))
    x = Dense(256, activation='relu')(state)
    x = concatenate([Flatten()(x), action])
    x = Dense(128, activation='relu')(x)
    out = Dense(1, activation='linear', kernel_initializer='random_uniform')(x)
    return Model([state, action], out)