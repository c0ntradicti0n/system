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


def infer(model_name, tree, valid_labels):
    keys, inp = tree.pull_lz(
        MODELS[model_name].config.batch_size, MODELS[model_name].config.n_samples
    )
    labels, score = MODELS[model_name].predict(inp)
    labels, score = list(
        labels.view(-1, MODELS[model_name].config.n_samples).tolist()
    ), list(score.view(-1, MODELS[model_name].config.n_samples).tolist())
    lsk = [
        (l, s, k)
        for l, s, k in zip(labels, score, keys)
        if all(i in list(l) for i in valid_labels)
    ]
    if not lsk:
        return []

    l_s_ks = list(
        list(zip(*sorted(zip(l, s, k), key=lambda x: x[0], reverse=True)))
        for l, s, k in lsk
    )

    return l_s_ks


def classifier(t):
    return infer("tas_3_only", t, [1, 2, 3])


def organizer(t):
    return infer("hierarchical_2", t, [1, 2])


control_score = 0.5


def minimax(t: Tree, i):

    added = False
    if random.choice(["|", "---"]) == "|":
        lsk = organizer(t)
        for l, s, k in lsk:
            added = "organizer"
            t.add_relation(k[1], k[0], "sub", h_score=s[0])
    else:
        lsk = classifier(t)
        for l, s, k in lsk:
            added = "synantithesis"

            t.add_relation(k[2], k[1], "ant", t_score=s[1], trident=i)
            t.add_relation(k[2], k[0], "syn", a_score=s[0], trident=i)
    if added:
        start = sorted(t.graph.degree, key=lambda x: x[1], reverse=True)[0][0]

        t.draw_graph(
            Tree.max_score_path(t.graph, start=start), root=start, path=f"{i}.png"
        )
        t.draw_graph_without_text_sequence(t.graph, path=f"{i}_without_text.png")
        t.save_state(i)

        print(f"{i} {added}")
        i += 1


# Initialize the PID controller
pid = PIDController(1, 0.1, 0.01)
threshold = 0.5  # Initial threshold for certainty score


not_done = True
# Get the inputs
inputs = get_inputs("tlp.txt")
T, i = Tree.load_state()
if not i:
    T, i = Tree(list(inputs.items())), 0


T.draw_graph()


while not_done:
    minimax(T, i)
    i+=1

