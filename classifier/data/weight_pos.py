from collections import Counter
from pprint import pprint

from classifier.data.triangle_pos import data_path, yield_from_file, ConceptPosition
from lib.json import decode, encode

print ("loading")
all_samples = []
n = 0
with open(data_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        # Deserialize the JSON string back into a tuple
        item = decode(line, ConceptPosition)
        all_samples.append(item)
        n += 1
print ("loaded")
c_labels = Counter([_[0][1] for _ in all_samples])
("counted")
print (c_labels)

relative_prob = {
    k: 1/(v/n)
    for k, v in
    c_labels.items()
}
pprint(relative_prob)

a = sum(relative_prob.values())
relative_prob = {
    k.name: v/a
    for k, v in
    relative_prob.items()
}
pprint(relative_prob)

print ([relative_prob.get(k, 0) for k in list(ConceptPosition.__members__)])