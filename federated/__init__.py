# federated/__init__.py
from .hfl import HorizontalFL, fedavg_aggregate
from .vfl import VerticalFL, XYZEncoder, RGBEncoder, ServerFusion

__all__ = [
    'HorizontalFL', 'fedavg_aggregate',
    'VerticalFL', 'XYZEncoder', 'RGBEncoder', 'ServerFusion'
]
