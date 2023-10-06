import random

from classifier.predict import MODELS
from integrator.reader import get_inputs
from integrator.tree import Tree


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def adjust_threshold(self, error):
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


def classifier(t):
    keys, inp = t.pull_lz(MODELS["tas_4_only"].config.n_samples)
    labels, score = MODELS["tas_4_only"].predict(inp)
    labells, score = list(labels.view(-1).tolist()), list(score.view(-1).tolist())
    if not all(i in list(labels) for i in [1, 2, 3]):

        return None, None

    labels = list(sorted(zip(keys, labells), key=lambda x: x[1]))
    return labels, score


def organizer(t):
    keys, inp = t.pull_lz(MODELS["hierarchical_2"].config.n_samples)
    labels, score = MODELS["hierarchical_2"].predict(inp)
    labels, score = list(labels.view(-1).tolist()), list(score.view(-1).tolist())
    if not all(i in list(labels) for i in [1,2]):

        return None, None

    labels = list(sorted(zip(keys, labels), key=lambda x: x[1]))
    return labels, score

control_score = 0.5

def minimax(t: Tree, pid):
    while True:
        added = False
        if random.choice(["|", "---"]) == "|":
            r, s = organizer(t)
            if r:
                added = True
                t.add_relation(r[0][0], r[1][0], "v", h_score = s[0])
        else:
            r, s = classifier(t)
            if r:
                added = True

                t.add_relation(r[0][0], r[1][0], "_t", t_score=s[0])
                t.add_relation(r[1][0], r[2][0], "_a", a_score=s[1])
                t.add_relation(r[2][0], r[0][0], "_s", s_score=s[2])
        if added:
            t.draw_graph(Tree.max_score_path(t.graph, "blub.png" ))
            print (".")


# Initialize the PID controller
pid = PIDController(1, 0.1, 0.01)
threshold = 0.5  # Initial threshold for certainty score


not_done = True
# Get the inputs
inputs = get_inputs("tlp.txt")
T = Tree(list(inputs.items())[:100])
T.draw_graph()


while not_done:

    # Use the minimax algorithm to find the best combinations
    best_element, best_value = minimax(T,  pid)

    # Update the tree based on the best value
    T.update_tree(T, best_element, score=best_value)
