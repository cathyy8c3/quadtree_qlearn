from qmaze import Qmaze, completion_check
from experience import *
from quadtree import leaves

def qtrain(model, maze, start, goal, **opt):
    global epsilon
    n_epoch = opt.get('n_epoch', 10)
    max_memory = opt.get('max_memory', 100000)
    data_size = opt.get('data_size', 100000)
    weights_file = opt.get('weights_file', "")
    name = opt.get('name', 'model')
    start_time = datetime.datetime.now()

    # If you want to continue training from a previous model,
    # just supply the h5 file name to weights_file option
    if weights_file:
        print("loading weights from file: %s" % (weights_file,))
        model.load_weights(weights_file)

    # Construct environment/game from numpy array: maze (see above)
    qmaze = Qmaze(maze, start, goal)

    # Initialize experience replay object
    experience = Experience(model, max_memory=max_memory)

    win_history = []  # history of win/lose game
    n_free_cells = len(qmaze.free_cells)
    hsize = qmaze.size // 2  # history window size
    win_rate = 0.0
    imctr = 1

    for epoch in range(n_epoch):
        # print("Starting training")
        loss = 0.0
        start_state = start
        qmaze.reset(start_state)
        game_over = False

        # get initial envstate (1d flattened canvas)
        envstate = qmaze.observe()

        n_episodes = 0
        while not game_over:
            # print("Playing...")
            valid_actions = qmaze.valid_actions()
            if not valid_actions: break
            prev_envstate = envstate
            # Get next action
            if np.random.rand() < epsilon:
                temp = []
                print("Exploring")
                for leaf1 in range(len(leaves)):
                    for leaf2 in valid_actions:
                        if leaves[leaf1].center() == leaf2.center():
                            temp.append(leaf1)
                action = random.choice(temp)
            else:
                print("Exploiting")
                temp = experience.predict(prev_envstate)
                print(model.predict(prev_envstate))
                action = np.argmax(temp)

            # Apply action, get reward and new envstate
            envstate, reward, game_status = qmaze.act(action)

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

            # Store episode (experience)
            episode = [prev_envstate, action, reward, envstate, game_over]
            experience.remember(episode)
            n_episodes += 1

            # Train neural network model
            inputs, targets = experience.get_data(data_size=data_size)
            h = model.fit(
                inputs,
                targets,
                epochs=8,
                batch_size=16,
                verbose=0,
            )
            loss = model.evaluate(inputs, targets, verbose=0)

        if len(win_history) > hsize:
            win_rate = sum(win_history[-hsize:]) / hsize

        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
        print(template.format(epoch, n_epoch - 1, loss, n_episodes, sum(win_history), win_rate, t))
        # we simply check if training has exhausted all free cells and if in all
        # cases the agent won
        if win_rate > 0.9: epsilon = 0.05
        # if sum(win_history[-hsize:]) == hsize and completion_check(model, qmaze):
        if sum(win_history[-hsize:]) == hsize:
            print("Reached 100%% win rate at epoch: %d" % (epoch,))
            break

    # Save trained model weights and architecture, this will be used by the visualization code
    h5file = name + ".h5"
    json_file = name + ".json"
    model.save_weights(h5file, overwrite=True)
    with open(json_file, "w") as outfile:
        json.dump(model.to_json(), outfile)
    end_time = datetime.datetime.now()
    dt = datetime.datetime.now() - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)
    print('files: %s, %s' % (h5file, json_file))
    print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (epoch, max_memory, data_size, t))
    return seconds


# This is a small utility for printing readable time strings:
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

def build_model(maze_size, lr=0.001):
    model = Sequential()
    model.add(Dense(maze_size, input_shape=(maze_size,)))
    model.add(PReLU())
    model.add(Dense(maze_size))
    model.add(PReLU())
    model.add(Dense(maze_size))
    model.compile(optimizer='adam', loss='mse')
    print("Model built")
    return model

