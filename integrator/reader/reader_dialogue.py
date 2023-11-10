from pprint import pprint

from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

# Define a grammar that can parse dialogues
grammar = Grammar(
    """
    text = (dialogue / other_line)*
    dialogue = speaker ws? line
    other_line = ~"[^\\n]*" ws?
    speaker = ~"[A-Z][a-z]*\\.?" ws
    line = ~".+"
    ws = ~"\\s+"
    """
)

class TextVisitor(NodeVisitor):
    speakers = {}

    def visit_text(self, node, visited_children):
        result = {}
        i = 1
        current_key = None

        for child in visited_children:
            child = [c for c in child if c]
            if not child:
                continue
            if len(child) == 1:
                if isinstance(child[0], dict):
                    current_key = i
                    result[i] = child[0]
                    i += 1
                elif current_key and child[0]:
                    result[current_key]["text"] += " " + child[0]
        return result

    def visit_dialogue(self, node, visited_children):
        speaker, _, line = visited_children
        # Expand the speaker's name if it's abbreviated
        full_speaker_name = self.speakers.get(speaker, speaker)
        return {'speaker': full_speaker_name, 'text': line}

    def visit_speaker(self, node, visited_children):
        speaker = node.text.strip().rstrip('.')
        # Assume the full name is spelled out at its first occurrence
        full_name = next((full_speaker_name for full_speaker_name in self.speakers.keys() if speaker in full_speaker_name), None)
        if not full_name:
            self.speakers[speaker] = speaker
            full_name = speaker
        return full_name

    def visit_line(self, node, visited_children):
        return node.text.strip()

    def visit_other_line(self, node, visited_children):
        return node.text.strip()

    def generic_visit(self, node, visited_children):
        return visited_children or node



def get_inputs(filename):
    with open(filename) as f:
        text = f.read()
    return parse_dialogue(text)

# Parse the dialogue text
def parse_dialogue(text):
    tree = grammar.parse(text)
    visitor = TextVisitor()
    result =  visitor.visit(tree)
    result = {k: f'[{v["speaker"]}] {v["text"]}' for k, v in result.items() if v}
    return result

if __name__ == "__main__":
    dialogue_text = """
    Socrates. My dear Phaedrus, whence come you, and whither are you going?
    Phaedrus. I come from Lysias the son of Cephalus, and I am going to
    take a walk outside the wall, 
    for I have been sitting with him the
    whole morning; and our common friend Acumenus tells me that it is
    much more refreshing to walk in the open air than to be shut up in
    a cloister.
    Soc. There he is right. Lysias then, I suppose, was in the town?
    """

    parsed_dialogue = parse_dialogue(dialogue_text)
    pprint(parsed_dialogue)

    pprint(get_inputs("texts/phaedrus.1b.txt"))

