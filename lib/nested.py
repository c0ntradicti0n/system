def extract_values(nested_dict):
    stack = [nested_dict]  # Initialize stack with the input dictionary
    result = []  # To store the extracted values

    while stack:
        current = stack.pop()  # Pop an element from the stack
        for key, value in current.items():
            if isinstance(value, dict):
                stack.append(
                    value
                )  # Add dictionary to the stack for further processing
            else:
                result.append(value)  # If it's a value, add it to the result list

    return result
