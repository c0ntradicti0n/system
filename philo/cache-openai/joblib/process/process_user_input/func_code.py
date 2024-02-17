# first line: 193
@memory.cache
def process_user_input(hint, instruction, prompt, task):
    api_result = llm(
        instruction=instruction.strip(),
        text=prompt + "\n\nThink also of " + hint,
        model=models[task],
    )
    output = api_result.choices[0].message.content

    print(output)
    return output
