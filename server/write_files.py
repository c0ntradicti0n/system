import os
from pprint import pprint

from server import config
from server.logic_generator import dump

text = """
          - topic: Knowledge Distillation
        1. Teacher Model
           In the process of knowledge distillation, a larger, more complex model (the "teacher") is used to guide the training of a smaller, simpler model (the "student").
        2. Simplification
           The goal of knowledge distillation is to simplify the complex patterns learned by the teacher model into a form that can be learned by the student model.
        3. Student Model
           The student model is the final output of the knowledge distillation process. It aims to mimic the teacher model's performance but with a simpler, more efficient structure.
        - self-antonym: Complex-Simplification
           The concept of 'Complex-Simplification' captures the paradox that within the simplification process in knowledge distillation, complex behaviors and understandings from the teacher model are still preserved and transferred to the student model.

   """
location = "33"
l = [t.strip() for t in text.replace("\n\n", "\n").split("\n") if t.strip()]
pprint(l)
topic, t1, tt1, t2, tt2, t3, tt3, aa, at = l
location_path = "/".join(location)

path = f"../{config.output_dir}/" + "/".join(location_path)
os.makedirs(
    f"{path}/.{topic.replace('- topic:', '').replace('topic:', '').strip()}",
    exist_ok=True,
)
dump(f"{path}/1-{t1.strip().replace('1. ', '')}.md", tt1)
dump(f"{path}/2-{t2.strip().replace('2. ', '')}.md", tt2)
dump(f"{path}/3-{t3.strip().replace('3. ', '')}.md", tt3)
dump(
    f"{path}/_{aa.replace('self-antonym:', '').replace('- ', '').strip()}.md", at
)  # result['explanation4'])
