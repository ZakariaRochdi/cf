# models/__init__.py
from .gin      import GIN
from .taconet  import TACONet
from .students import StudentGIN, StudentTACO, get_student

__all__ = ['GIN', 'TACONet', 'StudentGIN', 'StudentTACO', 'get_student', 'get_model']


def get_model(arch: str, num_classes: int = 8, **kwargs):
    """Factory pour les modèles enseignants."""
    arch = arch.lower()
    if arch == 'gin':
        return GIN(num_classes=num_classes, **kwargs)
    elif arch in ('taconet', 'taco'):
        return TACONet(num_classes=num_classes, **kwargs)
    raise ValueError(f"Architecture inconnue : {arch}. Choisir parmi : gin, taconet")
