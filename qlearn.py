from qtrain import build_model, qtrain
from quadtree import leaves
from mapgen import PASSABLE
from qmaze import floodfill

def run_qlearn(quadtree, start, goal):
    use_file = input("Do you want to load weights from an existing file? (y/n)\t")
    if use_file == "y":
        trained = True
    else:
        trained = False

    model = build_model(len(leaves))

    if not trained:
        count = 0
        for leaf in leaves:
            print("Leaf: " + str(count))
            count += 1
            if not leaf.color == PASSABLE:
                continue
            elif leaf.center() == start.center():
                continue
            elif not floodfill(start, leaf, quadtree):
                continue

            qtrain(model, quadtree, start, leaf, n_epoch=100, max_memory=8 * quadtree.count(), data_size=32)
    else:
        model.load_weights("model.h5")

    return model