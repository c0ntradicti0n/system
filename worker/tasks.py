import gc
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

    #old_graph = old_state.max_score_triangle_subgraph(
    #    old_state.graph, return_start_node=True, start_with_sub=False
    #)
    update_triangle_graph(old_state, i, hash_id)
    STATE = ITERATORS[hash_id]

    new_state, i = (
        old_state,
        i + 1,
    )
    states[hash_id] = new_state, i

    with catchtime("compute max score subgraph"):
        new_graph = new_state.max_score_triangle_subgraph(
            new_state.graph, return_start_node=True, start_with_sub=False
        )
    percentages = {
        str("_".join(kind)): iterator.get_percentage()
        for kind, iterator in new_state.iterators.items()
    }
    if STATE != "end":
        print(
            f"Computed: {STATE=} nodes={ len(new_graph[0].nodes)} edges={len(new_graph[0].edges)}"
            f" {str(percentages)=} {hash_id=} {i=}"
        )

    #try:
    #    patch = jsonpatch.make_patch(
    #        serialize_graph_to_structure(*old_graph),
    #        serialize_graph_to_structure(*new_graph),
    #    )
    #    serialized_patch = json.loads(patch.to_string())
    #except:
    #    print(f"error making patch {old_graph=} {new_graph=}")
    #    serialized_patch = []

    return {
        "state": serialize_graph_to_structure(*new_graph),
        "status": STATE,
        "i": i,
        "hash": hash_id,
        "percentages": percentages,
    }
