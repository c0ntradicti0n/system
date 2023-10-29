import gc
import json

import jsonpatch
from celery import Celery

from integrator.main import update_triangle_graph
from integrator.states import states
from integrator.tree import Tree

app = Celery("worker", broker="redis://redis:6379/1", backend="redis://redis:6379/1")


@app.task
def threerarchy(hash_id):
    gc.collect()

    old_state, i = states[hash_id]

    print(f"{len(old_state.graph.edges())=}")

    old_graph = Tree.max_score_triangle_subgraph(
        old_state.graph, return_start_node=True
    )
    new_graph = update_triangle_graph(old_state, i, hash_id, return_start_node=True)

    print(f"----------- **** { new_graph[0].nodes=}")
    print(f"----------- **** { new_graph[0].edges=}")

    new_state, i = (
        old_state,
        i + 1,
    )
    states[hash_id] = new_state, i

    print(f"MAKE PATCH FROM   ----------------------- {len(old_graph[0].edges())=}")
    print(f"MAKE PATCH FROM   ----------------------- {len(new_graph[0].edges())=}")
    print(f"MAKE PATCH FROM   ----------------------- {old_graph[1]=}")
    print(f"MAKE PATCH FROM   ----------------------- {new_graph[1]=}")

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
