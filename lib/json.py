import json
from enum import Enum

class EnumCodec(json.JSONEncoder):
    def __init__(self, enum_type, *args, **kwargs):
        self.enum_type = enum_type
        super().__init__(*args, **kwargs)

    def default(self, obj):
        if isinstance(obj, Enum):
            return {"__enum__": f"{obj.__class__.__name__}.{obj.name}"}
        return super().default(obj)

    @classmethod
    def decode(cls, enum_type):
        def decode_enum(dct):
            if "__enum__" in dct:
                enum_name, member_name = dct["__enum__"].split('.')
                if enum_name == enum_type.__name__:
                    return enum_type[member_name]
            return dct
        return decode_enum

def encode(enum_instance, enum_type):
    return json.dumps(enum_instance, cls=EnumCodec, enum_type=enum_type)

def decode(json_str, enum_type):
    return json.loads(json_str, object_hook=EnumCodec.decode(enum_type))
