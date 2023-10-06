from pprint import pprint

from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

grammar = Grammar(
    """
    text = (line / other_line)*
    line = ws? number ws? content ws?
    other_line = ~"[^\\n]*" ws?
    number = ~"[0-9.]+"
    content = ~"(?:(?!\\d+\\s*\\.).)*"  # Match any character until another numbered line or end of text
    ws = ~"\\s+"
    """
)


class TextVisitor(NodeVisitor):
    def visit_text(self, node, visited_children):
        result = {}
        current_key = None
        for child in visited_children:
            if isinstance(child, dict):
                current_key = list(child.keys())[0]
                result[current_key] = child[current_key]
            elif current_key and child:
                result[current_key] += " " + child
        return result

    def visit_line(self, node, visited_children):
        _, number, _, content, _ = visited_children
        return {number: content}

    def visit_other_line(self, node, visited_children):
        return node.text.strip()

    def visit_number(self, node, visited_children):
        return node.text.strip()

    def visit_content(self, node, visited_children):
        return node.text.strip()

    def generic_visit(self, node, visited_children):
        return visited_children and visited_children[0] or node


# Parse the text


# Parse the text
def parse_text(text):
    tree = grammar.parse(text)
    visitor = TextVisitor()
    return visitor.visit(tree)


def get_inputs(filename):
    with open(filename) as f:
        text = f.read()
    return parse_text(text)

if __name__ == "__main__":

    # Test
    text = """
    1 * 	Die Welt ist alles, was der Fall ist.
    OTHER TEXT
    1.1 	Die Welt ist die Gesamtheit der Tatsachen, nicht der Dinge.
    1.11 	Die Welt ist durch die Tatsachen bestimmt und dadurch, daß es alle Tatsachen sind.
    1.12 	Denn, die Gesamtheit der Tatsachen bestimmt, was der Fall ist und auch, was alles nicht der Fall ist.
    1.13 	Die Tatsachen im logischen Raum sind die Welt.
    1.2 	Die Welt zerfällt in Tatsachen.
    1.21 	Eines kann der Fall sein oder nicht der Fall sein und alles übrige gleich blieben.
    """

    parsed_data = parse_text(text)
    pprint(parsed_data)
