from qtrain import *
from quadtree import leaves

def run_qlearn(quadtree, start):
    use_file = input("Do you want to load weights from an existing file? (y/n)\t")
    if use_file == "y":
        trained = True
    else:
        trained = False

    qmaze = Qmaze(quadtree, start, start)
    model = build_model(qmaze)

    if not trained:
        count = 0
        for leaf in leaves:
            print("Leaf: " + str(count))
            count += 1
            if leaf.color == PASSABLE and not leaf.center() == start.center():
                qtrain(model, quadtree, start, leaf, epochs=1000, max_memory=8 * quadtree.count(), data_size=32)
    else:
        model.load_weights("model.h5")

    # print("Passable: ")
    # for leaf in leaves:
    #     print(leaf.color)

    return model