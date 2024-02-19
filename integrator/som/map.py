import itertools
from enum import Enum

class ConceptPosition(Enum):
    SUMMARIZING_CONCEPT = 0
    THESIS = 1
    ANTITHESIS = 2
    SYNTHESIS = 3
    LESS_COMPOUND = 4
    MORE_COMPOUND = 5
    SUBSUMED_INTO_THESIS = 6
    SUBSUMED_INTO_ANTITHESIS = 7
    SUBSUMED_INTO_SYNTHESIS = 8

class ConceptInsertionError(Exception):
    pass

# Placeholder for the actual prediction model
def predict_concept_position(concepts):
    # Preprocess the concepts list as required by your model
    model_output = models["som"].predict([concepts])

    # Convert model_output to ConceptPosition
    predicted_position = ConceptPosition(model_output)

    return predicted_position


def insert_concept_with_history(tree, path, concept, history):
    def navigate_and_create(tree, path):
        for p in path:
            tree = tree.setdefault(p, {})
        return tree

    def find_new_location_for_replaced_concept(tree):
        max_depth = 5
        possible_paths = [''.join(p) for i in range(1, max_depth + 1) for p in itertools.product('123', repeat=i)]
        for path in possible_paths:
            current_location = navigate_to_position(tree, path, create_if_missing=False)
            if current_location is None:
                return path
        raise ConceptInsertionError("No new location found for replaced concept.")

    def navigate_to_position(tree, path, create_if_missing=True):
        for p in path:
            if p not in tree and create_if_missing:
                tree[p] = {}
            tree = tree.get(p, {})
        return tree if tree else None

    while True:  # Loop to handle re-evaluation and re-insertion based on new predictions
        # Simulate getting the current context for the prediction (e.g., summarizing concept, thesis, etc.)
        context = [tree.get('.', '[NOT_SET]')] + [tree.get(str(i), {}).get('.', '[NOT_SET]') for i in range(1, 4)]
        prediction = predict_concept_position(context + [concept])

        if path in history:
            prev_prediction, _ = history[path]
            if prev_prediction != prediction:
                raise ConceptInsertionError(f"Contradicting predictions for path '{path}': {prev_prediction} vs {prediction}")
        history[path] = (prediction, concept)

        if prediction in {ConceptPosition.THESIS, ConceptPosition.ANTITHESIS, ConceptPosition.SYNTHESIS}:
            position = str(prediction.value)
            sub_tree = navigate_and_create(tree, path)
            if '.' in sub_tree.get(position, {}):
                replaced_concept = sub_tree[position]['.']
                new_location = find_new_location_for_replaced_concept(tree)
                path = new_location  # Update path for the new location
                continue  # Re-evaluate with the new path
            sub_tree[position] = {'.': concept}
            break
        elif prediction in {ConceptPosition.SUBSUMED_INTO_THESIS, ConceptPosition.SUBSUMED_INTO_ANTITHESIS, ConceptPosition.SUBSUMED_INTO_SYNTHESIS}:
            new_path = path + str(prediction.value - 5)
            path = new_path  # Update path for subsumption
            continue  # Re-evaluate with the new path
        elif prediction == ConceptPosition.SUMMARIZING_CONCEPT:
            if path:
                navigate_and_create(tree, path[:-1])[path[-1]] = {'.': concept}
            else:
                tree['.'] = concept
            break
        else:
            raise NotImplementedError(f"Handling for prediction {prediction} not implemented.")

    return tree