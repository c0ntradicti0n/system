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
6.1265 Logic can always be conceived to be such that every proposition is its own proof.


6.127 All propositions of logic are of equal rank; there are not some which are essentially primitive and others deduced from these.

Every tautology itself shows that it is a tautology.


6.1271 It is clear that the number of "primitive propositions of logic" is arbitrary, for we could deduce logic from one primitive proposition by simply forming, for example, the logical product of Frege's primitive propositions. (Frege would perhaps say that this would no longer be immediately self-evident. But it is remarkable that so exact a thinker as Frege should have appealed to the degree of self-evidence as the criterion of a logical proposition.)


6.13 Logic is not a theory but a reflexion of the world.
    """

    parsed_data = parse_text(text)
    pprint(parsed_data)
