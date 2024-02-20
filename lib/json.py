import json
from enum import Enum
class EnumCodec(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return {"__enum__": f"{obj.__class__.__name__}.{obj.name}"}
        else:
            return super().default(obj)

    @staticmethod
    def decode_enum(dct, enum_type=None):
        if "__enum__" in dct:
            enum_name, member_name = dct["__enum__"].split('.')
            # Assuming enum_type is provided and matches enum_name
            if enum_type and enum_type.__name__ == enum_name:
                return enum_type[member_name]
        return dct

def encode(data, enum_type=None):
    # Convert enum keys to strings
    if isinstance(data, dict):
        data = {k.name if isinstance(k, Enum) else k: v for k, v in data.items()}
    return json.dumps(data, cls=EnumCodec)

def decode(json_str, enum_type):
    object_hook = lambda dct: EnumCodec.decode_enum(dct, enum_type=enum_type)
    return json.loads(json_str, object_hook=object_hook)