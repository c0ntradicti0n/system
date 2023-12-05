import gc
import logging

from celery import Celery

from integrator.main import ITERATORS, update_triangle_graph
from integrator.serialize import serialize_graph_to_structure
from integrator.states import states
from lib.t import catchtime

app = Celery("worker", broker="redis://redis:6379/1", backend="redis://redis:6379/1")


@app.task
def threerarchy(hash_id):
    gc.collect()

    old_state, i = states[hash_id]

    print(f"task {hash_id} {i}")

    # old_graph = old_state.max_score_triangle_subgraph(
    #    old_state.graph, return_start_node=True, start_with_sub=False
    # )
    update_triangle_graph(old_state, i, hash_id)

    STATE = ITERATORS[hash_id]

    new_state, i = (
        old_state,
        i + 1,
    )
    states[hash_id] = new_state, i

    if "syn_1_syn_2" in new_state.progress():
        with catchtime("compute max score subgraph"):
            new_graph = new_state.max_score_triangle_subgraph(
                new_state.graph, return_start_node=True, start_with_sub=False
            )
        with catchtime("serialize"):
            serialized = serialize_graph_to_structure(*new_graph)
    else:
        logging.info("Skipped graph, no results possible")
        serialized = {}
    progress = new_state.progress()

    return {
        "state": serialized,
        "status": STATE,
        "i": i,
        "hash": hash_id,
        "progress": progress,
    }
