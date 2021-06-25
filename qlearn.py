from qtrain import *

def run_qlearn(quadtree, start, goal):
    qmaze = Qmaze(quadtree, start, goal)
    model = build_model(qmaze)
    qtrain(model, quadtree, start, goal, epochs=1000, max_memory=8 * quadtree.count(), data_size=32)