# `System` - ChatGPT`s understanding of Hegels dialectic

I instructed ChatGPT to continuously update a data-structure to imitate [G.W.F. Hegel](https://en.wikipedia.org/wiki/Georg_Wilhelm_Friedrich_Hegel)

You can visit the result on [https://polarity.science](https://polarity.science)

It visualizes in the style of [https://hegel-system.de/en/](https://hegel-system.de/en/) the basic intention of Hegels dialectic:
Fractalize all concepts into triples of thesis, antithesis and synthesis.

Hegel thought his method is more accurate than mathematics, because it is more general and can be applied to all topics.
We don't know if he was right, but we can try to apply his method to all topics and see what happens, so it is a bit like rubber.


It did not write a reasonable README, so I just show, what is the prompt and it understood it.

```

You are extending a dialectical system, emulating Hegel's methodology, where concepts unfold within a fractal structure of triples. Each triple consists of:

 - Thesis (1)
 - Antithesis (2)
 - Synthesis (3)
 
Additionally we mention about each triple:
 - Topic (.) 
 - Inversive Dialectical Antonym (This is a pivot concept that amplifies the evolution of the argumentation by identifying the conflict within the dialectic triple, thereby triggering the formation of the next triple. This self-applicable antonym to the thesis expresses ideas like 'minus * minus = plus', or 'the disappearance of disappearance is existence.') (_)

You'll work with a truncated JSON reflecting your current progress. Your goal is to enhance and enrich the thematic structure of the triples within this JSON and also diving deeper by creating new tripples.

Detect json-paths where the JSON lacks cohesion or displays incorrect structure. Propose improvements by modifying titles, or introducing new elements. Use exactly (!) valid json-path syntax below.

Using this syntax we can state that, if y is in the values [1, 2, 3]:
  $.[x1...xn][y]['.']
is at the same level of explanation as  
  $.[x1...xn][y]
except, that in the upper case the semantic and fractalic explanation goes one level deeper into the thing than in the lower.

Every nesting level should have such a '.'-entry for telling the topic and also a '_'-entry to mark dialectical conceptual movement. 

Examples:

 - replace a topic title: 
$.['1']['3']['2']['.'] "new topic title"

 - offer a new inversive antonym: 
$.['3']['2']['1']['_'] "new antonym"
    
 - suggest an alternative antithesis: 
$.['1']['2']['1'] "new thesis"

 - suggest an alternative antithesis: 
$.['1']['2']['2'] "new antithesis"

 - propose a synthesis: 
$.['1']['3']['2']['3'] "new synthesis"

 - change bigger parts:
$.['3']['1'] {"1": "Force", "2": "Motion", "3": "Energy", "_": "Inertia-Dynamics", ".": "Physics"}
$.['3']['2'] {"1": "Cell", "2": "Organism", "3": "Ecosystem", "_": "Individual-Community", ".": "Biology"}
$.['3']['1'] {"1": "Atom", "2": "Molecule", "3": "Compound", "_": "Element-Compound", ".": "Chemistry"}
    
Stick exactly to this syntax, don't exchange: '.', '_antonym', '1', '2', '3' with more meaningful information, put the meaning only into the titles.
Respond only with json-paths and succinct suggestions. Avoid any other explanatory phrase. Your proposals should be limited to 1-4 word titles, no sentences.
Thus your output should be a list with a syntax like this:

$.['1']['1']['1']['.'] "Abstraction"
$.['1']['1']['.'] "Logic"
$.['1']['1']['_'] "Minus-minus-plus"
$.['2']['3']['1'] "Morality"
$.['2']['3']['2'] {"1": "major", "2": "minor", "3": "sixth-chord", "_": "diminished", ".": "Musical harmony"}


Don't be audacious and dont just change the series of words in one title, keep it as simple as possible, avoid repetitions in titles, the simplest words are the best and less words is more content. Be enourmously precise with your answers and the json-paths, every wrong json-path causes chaos and will kill the whole project.
Focus only on scientific objective topics as math, geometry, physics, chemistry, biology, epistemology, music, colors, linguistics and other real things with popular polarities. Absolutely avoid any topics of philosophy of mind and psychology and avoid the topic of consciousness at all. Philosophers nowadays are not able to think about consciousness, the language is partying there too much.
Focus on a top-down approach to get to some more dialectical structure; first all upper levels, then the lower levels.
Focus on completeness of the fractal, please fill up all uncomplete triples, rather add new triples than improving existing ones.

This time put the explanation of minus*minus=plus and minus-minus=plus and three times turning left is right and vice versa into the correct place.
```

So, write me the README ok?