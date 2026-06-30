from .nodes import NODE_CLASS_MAPPINGS as _N1, NODE_DISPLAY_NAME_MAPPINGS as _D1
from .infinite_sampler import NODE_CLASS_MAPPINGS as _N2, NODE_DISPLAY_NAME_MAPPINGS as _D2
from .infinite_sampler_v3 import NODE_CLASS_MAPPINGS as _N3, NODE_DISPLAY_NAME_MAPPINGS as _D3
from .infinite_sampler_v4 import NODE_CLASS_MAPPINGS as _N4, NODE_DISPLAY_NAME_MAPPINGS as _D4
from .infinite_sampler_v5 import NODE_CLASS_MAPPINGS as _N5, NODE_DISPLAY_NAME_MAPPINGS as _D5
from .infinite_sampler_v6 import NODE_CLASS_MAPPINGS as _N6, NODE_DISPLAY_NAME_MAPPINGS as _D6
from .infinite_sampler_v7 import NODE_CLASS_MAPPINGS as _N7, NODE_DISPLAY_NAME_MAPPINGS as _D7
from .muse_director_v1 import NODE_CLASS_MAPPINGS as _NM1, NODE_DISPLAY_NAME_MAPPINGS as _DM1

NODE_CLASS_MAPPINGS = {**_N1, **_N2, **_N3, **_N4, **_N5, **_N6, **_N7, **_NM1}
NODE_DISPLAY_NAME_MAPPINGS = {**_D1, **_D2, **_D3, **_D4, **_D5, **_D6, **_D7, **_DM1}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
