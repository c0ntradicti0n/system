from integrator.reader import get_inputs


class TripletNode:
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, child_node):
        if len(self.children) < 3:
            self.children.append(child_node)


class TripletTree:
    def __init__(self):
        self.root = None

    def insert(self, value, parent=None):
        new_node = TripletNode(value)
        if not self.root:
            self.root = new_node
        elif parent:
            parent.add_child(new_node)
        return new_node


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


def classifier(triplet1, triplet2):
    # Call the AI model to classify the relation and get the certainty score
    relation, certainty_score = AI_MODEL_1.classify(triplet1, triplet2)
    return relation, certainty_score


def organizer(inputs):
    # Call the AI model to organize the inputs into a triplet and get the certainty score
    triplet, certainty_score = AI_MODEL_2.organize(inputs)
    return triplet, certainty_score


def minimax(node, depth, is_maximizer, alpha, beta, threshold):
    if depth == 0 or is_terminal(node):
        return evaluate(node)

    if is_maximizer:
        max_eval = float("-inf")
        for child in node.children:
            eval = minimax(child, depth - 1, False, alpha, beta, threshold)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float("inf")
        for child in node.children:
            eval = minimax(child, depth - 1, True, alpha, beta, threshold)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def evaluate(node):
    # Evaluate the node based on the certainty score and other criteria
    return node.value.certainty_score


def is_terminal(node):
    # Check if the node is a terminal node
    return len(node.children) == 0


# Initialize the PID controller
pid = PIDController(1, 0.1, 0.01)
threshold = 0.5  # Initial threshold for certainty score


not_done = True
tree = TripletTree()


while not_done:
    # Get the inputs
    inputs = get_inputs("tlp.txt")

    # Organize the inputs into a triplet
    triplet, certainty_score = organizer(inputs)

    # If the certainty score is below the threshold, try more classifications
    while certainty_score < threshold:
        # Adjust the threshold using the PID controller
        error = threshold - certainty_score
        threshold += pid.adjust_threshold(error)

        # Try more classifications
        triplet, certainty_score = organizer(inputs)

    # Insert the triplet into the tree
    tree.insert(triplet)

    # Use the minimax algorithm to find the best combinations
    best_value = minimax(tree.root, 3, True, float("-inf"), float("inf"), threshold)

    # Update the tree based on the best value
    update_tree(tree, best_value)
