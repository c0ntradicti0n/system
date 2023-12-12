def flatten_dict_values(d):
    def flatten(item):
        if isinstance(item, dict):
            for value in item.values():
                yield from flatten(value)
        else:
            yield item  #

    return list(flatten(d))
