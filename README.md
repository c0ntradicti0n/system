# `System` - LLM - dialectics
## `FractalVerse` - a fractalized verse of the universe


I instructed ChatGPT to continuously update a data-structure to imitate [G.W.F. Hegel](https://en.wikipedia.org/wiki/Georg_Wilhelm_Friedrich_Hegel)

You can visit the result on [https://polarity.science](https://polarity.science)

It visualizes in the style of [https://hegel-system.de/en/](https://hegel-system.de/en/) the basic intention of Hegels dialectic:
Fractalize all concepts into triples of thesis, antithesis and synthesis.

Hegel thought his method is more accurate than mathematics, because it is more general and can be applied to all topics.
We don't know if he was right, but we can try to apply his method to all topics and see what happens, so it is a bit like rubber.


It did not write a reasonable README, so I just show, what is the prompt and it understood it.

```
You are extending a dialectical system, emulating Hegel's methodology, where concepts unfold within a 
fractal structure of triples. Each triple consists of:

 - Thesis (1)
 - Antithesis (2)
 - Synthesis (3)

Each triple is a dialectical unit, which can be further extended by adding a new triple to any of them.

Additionally we explain about each triple a antinomic mutual relation. Remember the Kantian antinomies, and 
other famous intricacies. 
This should point to  the evolution of the argumentation by identifying the conflict within the 
dialectic triple, thereby triggering the formation of the next triple. This self-applicable antonym to the thesis 
expresses ideas like 'minus * minus = plus', or 'the disappearance of disappearance is existence.') (_)

There is also this constraint: 
The fractal lives from semantic analogies, every subtopic from [1-3]*1 should mirror the topics in [1-3]*2 by 
ANALOGY (as far as possible). Please respect this, therefore you are given example other entries, that might correspond.

Respect, that it moves from the most abstract to the most concrete concepts when moving deeper into the nesting 
and horizontally it moves by the defining other concepts by first treating the ingredients and then the complexities 
of those. Another Note on categorization: The fractal has to conquer the realm of overlapping categories. Feelings might be
 "happy" and "sad" as bigger categories and "joyful" and "depressed" might appear as sub-categories or 
 rather side-categories, this is out preferred approach.
 This means instead of 
 
11 happy
111 joyful
12 sad
121 depressed

we would like to organize it like this:

111 happy
112 sad
121 joyful
122 depressed

So the most basic category opens the field and gets mirrored by the other categories and dividing up all opposites 
inside a nesting.

Also respect, that we handle the concepts as "ideals" and daily politics and doubts are not the topic here, if something
needs to be changed, it should be changed, there must not be a hint, that something might not be right.

You'll work with a representation reflecting your current progress. Each unit is addressed via a key as "13221." 
for the thesis in "1322".
Your goal is enrich the thematic structure of the fractal triples by diving deeper by creating 
new triples.

Every nesting level should have a bare path for telling the theme and also a '_'-entry to mark dialectical conceptual 
movement, every path should match this regex: [1-3]_?

Examples:

- suggest alternative topic
3331. new topic

- suggest alternative inversion antonym
332_ new inversion antonym

- suggest alternative thesis
31 new thesis

- suggest alternative antithesis
312322 new antithesis

- suggest alternative synthesis
2233123 new synthesis

- add multiple things like (but with other contents). PREFER THIS FOR MANY ENTRIES
31 {1: "Force", 2: "Motion", 3: "Energy", "_": "Inertia-Dynamics", ".": "Physics"}
32 {1: "Cell", 2: "Organism", 3: "Ecosystem", "_": "Individual-Community", ".": "Biology"}
31 {1: "Atom", 2: "Molecule", 3: "Compound", "_": "Element-Compound", ".": "Chemistry"}

Stick exactly to this syntax, don't exchange: '.', '_' with more meaningful information, put the meaning only into 
the strings.
Respond only with the keys like 31332 and succinct suggestions. Avoid any other explanatory phrase! Your proposals should 
be limited to 1-4 word titles, no sentences.
Thus your output should be a list with a syntax like this:

3331 "Colors"
332 "Wave-Particle"
3 "Light"
31232 "Darkness"
223312 "Color"
31 {1: "Force", 2: "Motion", 3: "Energy", "_": "Inertia-Dynamics", ".": "Physics"}

Don't be audacious and dont just change the series of words in one title, keep it as simple as possible, avoid 
repetitions in titles, the simplest words are the best and less words is more content. Be enormously precise 
with your answers and the keys, every wrong path causes chaos and will kill the whole project.
Focus only on scientific objective topics as math, geometry, physics, chemistry, biology, epistemology, music, 
colors, linguistics and other real things with popular polarities. Absolutely avoid any topics of philosophy 
of mind and psychology and avoid the topic of consciousness at all. Philosophers nowadays are not able to think 
about consciousness, the language is partying there too much.
Focus on a top-down approach to get to some more systematic dialectical structure; first all upper levels, then the lower levels.
Focus on completeness of the fractal, please fill up all incomplete triples, rather add new triples than improving existing ones.



And please dump all your knowledge accurately about 13121, Reaction,

Here is a truncated version of the current dialectical system to the path, where you should operate on:

 System
_ System-Chaos
1 Objectivity
1_ Synthetic a Priori
11 Things and Properties
11_ Intrinsic-Extrinsic
12 Framework of Placement
12_ Dimensional Progression
13 Nature
13_ Whole-Parts
131 Nature Mechanics
131_ Conservation-Transformation
1311 Physics
1311_ Energy-Matter
13111_ Field Dynamics-Constituency
13112 Nuclear Physics
13112_ Stability-Instability
13113 Mechanics
13113_ Determinism-Indeterminism
1312 Chemistry
1312_ Element-Compound
13121 Reaction
1313 Biology
1313_ Static-Dynamic
13131 Life
132 Earth
132_ Solid-Liquid-Gas
133 ' Astronautics'
133_ Exploration-Settlement
2 Subjectivity
2_ Objective-Subjective
21 Natrual Necessity
21_ Fatalism-Free will
22 Values
22_ Revaluation of all values
23 Making
23_ Invention-Adaptation
231 Domains of Knowledge
231_ Abundance of Knowledge
2311 Art
2311_ Creator-Creation
23111 Innovation
23111_ Discovery-Invention
23112 Judgments
23112_ Standard-Deviation
23113 Responsibilities
23113_ Duty-Freedom
2312 Science of Science
2312_ Methodology-Epistemology
23121 Scientific Method
23121_ Hypothesis-Proof
23122 Paradigm Shifts
23122_ Static-Dynamic
23123 Measurement
23123_ Qualitative-Quantitative
2313 Philosophy of Science
2313_ Model-Reality
23131 Scientific Realism
23131_ Observable-Unobservable
23132 Causality
23132_ Cause-Effect
23133 Falsifiability
23133_ Testable-Untestable
3 Culture
3_ Negotiation-of-Meaning
31 Usage
31_ Aim-Effect
32 Institutional Structures
32_ Conservative-Revolutionary
33 Cultural Embodiment
33_ Tradition-Innovation
331 Religion
331_ Divine-Mundane
3311 Theism
3311_ Monotheism-Polytheism
33111 Christianity
33111_ Trinity-Oneness
33112 Islam
33112_ Sunnism-Shiism
33113 Hinduism
33113_ Deity-Atman
3312 Religious Worldviews
3312_ Monotheism-Polytheism
33121 Monotheistic Faiths
33121_ God-Creation
33122 Polytheistic Pantheon
33122_ Gods-Interactions
33123 Animism
33123_ Spirit-Nature
3313 Atheism
3313_ Belief-Disbelief
33131 Rational Atheism
33131_ Evidence-Absence
33132 Existential Atheism
33132_ Meaning-Nihilism
33133 Agnosticism
33133_ Known-Unknown



And, please dump all you wisdom as a table of contents (only provide titles for chapters) by using our number format accurately and deeply nested about 
 
13121, Reaction,


Respect that it should work on analogies resembling the following topics

1311_ Energy-Matter
1311. Physics
13112 Nuclear Physics
13113 Mechanics
    
and don't output any other commentary, the code will extract the titles from your output.
  
Can you please stay on the ground and first develop the topics with quite normal knowledge? As motion is a kind combination of space and time.
Another good hint to the structure, that MUST be observed, is: 1 - is the general, 2 - is the particular, 3 - is the individual as their combination.
```

# TODO

 * [ ] add github pages
 * [ ] fix links in tooltips
 * [ ] debug more scoring
 * [ ] add dialog-guide to organize your own topics
 * [ ] debug socket.io - fe and be
 * [ ] add online-learning for models
 * [ ] save and apply edit actions
 * [ ] add life-inspection of training
 * [ ] make the titels appear in front of other triangles
 * [ ] mobile layout
 * [ ] run in kube


# DONE
