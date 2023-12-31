import logging

try:
    import torch
except ImportError:
    torch = None
    logging.warning(
        "torch not found. Install PyTorch to enable shape inference and tensor support."
    )


def flatten(lst):
    """Flatten a nested list."""
    for el in lst:
        if isinstance(el, (list, tuple)):
            yield from flatten(el)
        else:
            yield el


def build_nested_list(flat_list, shape):
    """Build a nested list from a flattened list and a shape."""
    if len(shape) == 1:
        return flat_list[: shape[0]]
    else:
        split = len(flat_list) // shape[0]
        return [
            build_nested_list(flat_list[i * split : (i + 1) * split], shape[1:])
            for i in range(shape[0])
        ]


def view_shape(lst, shape):
    """Reshape a nested list."""
    flat_list = list(flatten(lst))
    total_elements = len(flat_list)

    # Check for negative indices and compute the inferred dimension
    if shape.count(-1) > 1:
        raise ValueError("Only one dimension can be inferred.")
    if -1 in shape:
        inferred_dim = total_elements // (-product(shape))
        shape = [inferred_dim if dim == -1 else dim for dim in shape]

    if total_elements != product(shape):
        raise ValueError(
            f"Shape doesn't match the total number of elements in the list."
            f"Expected {product(shape)} but got {total_elements}."
        )

    return build_nested_list(flat_list, shape)


def get_shape(lst):
    """Return the shape of a nested list."""
    if isinstance(lst, (list, tuple, torch.Tensor)):
        return [len(lst)] + get_shape(lst[0])
    else:
        return []


def product(lst):
    """Return the product of a list of numbers."""
    result = 1
    for num in lst:
        result *= num
    return result


def to_tensor(tensor_lists):
    # Check if we are at the deepest level (i.e., the current element is a tensor)
    if isinstance(tensor_lists[0], torch.Tensor):
        return torch.stack(tensor_lists)

    # Otherwise, recursively process each sublist
    stacked_sublists = [to_tensor(sublist) for sublist in tensor_lists]

    # Stack the resulting tensors
    return torch.stack(stacked_sublists)


if __name__ == "__main__":
    lst = [[["a", "b"], ["c", "d"]], [["e", "f"], ["g", "h"]]]

    new_shape = [2, 4]
    reshaped_lst = view_shape(lst, new_shape)
    print(reshaped_lst)  # [
