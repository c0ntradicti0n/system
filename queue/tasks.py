import json

import jsonpatch
from celery import Celery

from integrator.states import states
from integrator.main import update_triangle_graph
from integrator.tree import Tree

app = Celery('tasks', broker='redis://redis:6379/0', backend='redis://redis:6379/0')

@app.task
def threerarchy(hash_id):
    old_state, i = states[hash_id]

    old_graph = Tree.max_score_triangle_subgraph(
        old_state.graph, return_start_node=True
    )

    new_graph = Tree.max_score_triangle_subgraph(
        update_triangle_graph(old_state, i, hash_id), return_start_node=True
    )

    new_state, i = (
        old_state,
        i + 1,
    )
    states[hash_id] = new_state, i

    try:
        patch = jsonpatch.make_patch(
            Tree.serialize_graph_to_structure(*old_graph),
            Tree.serialize_graph_to_structure(*new_graph),
        )
        serialized_patch = json.loads(patch.to_string())
    except:
        print(f"error making patch {old_graph=} {new_graph=}")
        serialized_patch = []

    return serialized_patch
