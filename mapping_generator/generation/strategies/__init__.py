# mapping_generator/generation/strategies/__init__.py

"""
Módulo de estratégias de dificuldade para geração CGRA.

Estratégias disponíveis:
- SystematicStrategy: Dificuldade fixa e determinística
- RandomStrategy: Dificuldade aleatória com adaptação
"""

from .systematic import SystematicStrategy
from .random_strategy import RandomStrategy

__all__ = ['SystematicStrategy', 'RandomStrategy']
