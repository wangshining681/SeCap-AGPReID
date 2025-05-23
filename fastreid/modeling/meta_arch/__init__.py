# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .build import META_ARCH_REGISTRY, build_model


# import all the meta_arch, so they will be registered
from .baseline import Baseline
from .mgn import MGN
from .pcb import PCB
from .moco import MoCo
from .distiller import Distiller
from .baseline_multi_view import Baseline_multiview
from .baseline_multi_view_SeCap import Baseline_multiview_SeCap
