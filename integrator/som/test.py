import pytest
from integrator.som.map import insert_concept_with_history, ConceptPosition  # Adjust import path as needed


@pytest.mark.parametrize("insertions,expected_structure", [
    # Test Case 1: Philosophical Concepts
    ([
         ('', 'To Be', ConceptPosition.SUMMARIZING_CONCEPT),
         ('1', 'Not To Be', ConceptPosition.ANTITHESIS),
         ('1', 'Become', ConceptPosition.SYNTHESIS)
     ],
     {
         '.': 'To Be',
         '1': {'.': 'Not To Be'},
         '3': {'.': 'Become'}
     }),

    # Test Case 2: Mathematical Operations: Plus, Minus, Zero
    ([
         ('', 'Plus', ConceptPosition.SUMMARIZING_CONCEPT),
         ('1', 'Minus', ConceptPosition.ANTITHESIS),
         ('1', 'Zero', ConceptPosition.SYNTHESIS)
     ],
     {
         '.': 'Plus',
         '2': {'.': 'Minus'},
         '3': {'.': 'Zero'}
     }),

    # Additional test cases follow the same structure...
])
def test_insert_concepts(insertions, expected_structure):
    tree, history = {}, {}  # Initialize an empty tree and history for each test run
    for path, concept, prediction in insertions:
        tree = insert_concept_with_history(tree, path, concept, prediction, history)

    assert tree == expected_structure
