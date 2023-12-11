from pprint import pprint

import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def parse_sentences(text):
    # Use spaCy's pipeline for sentence segmentation
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences

def get_inputs(filename):
    with open(filename, 'r') as f:
        text = f.read()
    return parse_sentences(text)

if __name__ == "__main__":
    dialogue_text = """
    My dear Phaedrus, whence come you, and whither are you going?
    I come from Lysias the son of Cephalus, and I am going to
    take a walk outside the wall, 
    for I have been sitting with him the
    whole morning; and our common friend Acumenus tells me that it is
    much more refreshing to walk in the open air than to be shut up in
    a cloister.
    There he is right. Lysias then, I suppose, was in the town?
    """

    parsed_dialogue = parse_sentences(dialogue_text)
    pprint(parsed_dialogue)

    # Example of reading from a file
    # pprint(get_inputs("texts/phaedrus.1b.txt"))
