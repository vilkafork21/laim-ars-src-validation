from typing                 import Mapping
from types                  import MappingProxyType
from dataclasses            import dataclass

from pathlib                import PurePath

from codebase_node_compose  import Composition

from codebase_node          import Pack


@dataclass(frozen = True)
class Package:
    pack_method         : Composition       = Pack.string

    base                : PurePath          = PurePath(__file__).parent / 'data'

    workers             : Mapping[str, str] = MappingProxyType({
        'validation'    : 'laim-ars-validation',
    })

    exclude_dirs        : frozenset[str]    = frozenset({'.git', '.venv', '__pycache__'})
    exclude_suffixes    : frozenset[str]    = frozenset({'.pyc', '.pyo'})