import regex as re

def remove_links(text, pattern= re.compile(r"\[(?P<text>[^\]]+)\]\(([^)]+)\)")):
    return pattern.sub(r"\g<text>", text)
