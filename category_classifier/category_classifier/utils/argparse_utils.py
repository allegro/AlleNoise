from enum import Enum


class ArgParseEnum(Enum):
    def __str__(self) -> str:
        return self.name.lower()

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def argparse(cls, arg_name: str):
        try:
            return cls[arg_name.upper()]
        except KeyError:
            return arg_name
