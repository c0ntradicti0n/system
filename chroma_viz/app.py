import os

from classifier.result.predict import MODELS
from lib.proximity import set_up_db_from_model

input_dict = dict(
    enumerate(
        list(
            set(
                [
                    *"""

I am running Django on Heroku with zero-downtime feature. This means that during deployment there are two version of code running (old and new) on the same database. That's why we need to avoid any backward incompatible migrations.

It there a possibility to exclude a field from Django query on a given model?

Let say we have a model (version 1):

class Person(models.Model):
    name = models.CharField()
    address = models.TextField()
In some time in the future we want to move address to the separate table. We know that we should not delete a field for older code to work so Person model may look like (version 2):

class Person(models.Model):
    name = models.CharField()
    address = models.ForeignKey(Address)
    _address = models.TextField(db_name='address')
This way if old code will query for address it will get it from Person table even if database has been migrated (it will be an old value, but let assume thats not a big issue).

How now I can safetly delete _address field? If we will deploy version 3 with _address field deleted then code for version 2 will still try to fetch _address on select, even if it's not used anywhere and will fail with "No such column" exception.

Is there a way to prevent this and mark some field as "non-fetchable" within the code for version 2? So version 2 will not delete field, but will not fetch it anymore and version 3 will delete field.""".split(),
                    "Some text",
                    "Some more text",
                    "Even more text" "plus",
                    "minus",
                    "times",
                    "divided by",
                    "equals",
                    "root",
                    "green",
                    "blue",
                    "red",
                    "yellow",
                    "orange",
                    "purple",
                    "pink",
                    "black",
                    "white",
                    "hello",
                    "goodbye",
                    "yes",
                    "no",
                    "maybe",
                    "always",
                    "never",
                    "sometimes",
                    "up",
                    "down",
                    "left",
                    "right",
                    "forward",
                    "backward",
                    "north",
                    "south",
                    "east",
                    "west",
                    "northwest",
                    "northeast",
                    "southwest",
                    "southeast",
                    "northwest",
                    "northeast",
                    "southwest",
                    "southeast",
                    "northwest",
                    "blood",
                    "sweat",
                    "tears",
                    "pain",
                    "suffering",
                    "joy",
                    "happiness",
                    "sadness",
                    "anger",
                    "fear",
                    "love",
                    "hate",
                    "friendship",
                    "family",
                    "brother",
                    "sister",
                    "mother",
                    "father",
                    "son",
                    "daughter",
                    "grandfather",
                    "grandmother",
                    "grandson",
                    "granddaughter",
                    "aunt",
                    "uncle",
                    "cousin",
                    "nephew",
                    "niece",
                    "husband",
                    "wife",
                    "boyfriend",
                    "girlfriend",
                    "partner",
                    "friend",
                    "enemy",
                    "rival",
                    "acquaintance",
                    "stranger",
                    "neighbor",
                    "classmate",
                    "teacher",
                    "student",
                    "doctor",
                    "nurse",
                    "scientist",
                    "engineer",
                    "programmer",
                    "artist",
                    "musician",
                    "writer",
                    "poet",
                    "actor",
                    "actress",
                ]
            )
        )
    )
)

model_name = "opposites"
model = MODELS[model_name]
config = MODELS[model_name].config

vector_store = set_up_db_from_model("xxx", input_dict, model, config)
from chromaviz import visualize_collection

print(vector_store.search("hello", k=4, search_type="mmr"))


# This example only specifies a relevant query
visualize_collection(vector_store._collection)
