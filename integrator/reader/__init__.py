from .reader_dialogue import parse_dialogue
from .reader_enumerated import parse_enumerated
from .reader_sentences import parse_sentences


def parse_text(text):

    result = parse_enumerated(text)

    if not result:
        result = parse_sentences(text)
    if not result:
        result = parse_dialogue(text)

    return result


def get_inputs(filename):
    with open(filename) as f:
        text = f.read()
    return parse_dialogue(text)


if __name__ == "__main__":

    from pprint import pprint

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

    pprint(get_inputs("texts/euclid.txt"))
    pprint(get_inputs("texts/phaedrus.1b.txt"))
