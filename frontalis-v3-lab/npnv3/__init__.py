"""npnv3: 経験で人格が形成される外部ニューラルネットワーク（NPN）の最小実装 v3.3（第 1 部: NN の再検証と消去）。"""
from .config import Config
from .lexicon import Lexicon
from .npn import NPN, StateIntegrityError, StateVersionError
from .schema import SchemaError

__all__ = ["NPN", "Config", "Lexicon", "SchemaError", "StateIntegrityError", "StateVersionError"]
__version__ = "3.3.0"
