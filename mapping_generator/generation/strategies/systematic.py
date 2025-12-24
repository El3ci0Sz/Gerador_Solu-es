# mapping_generator/generation/strategies/systematic.py

import logging
from typing import Tuple, Dict, Optional, Union
from .base import DifficultyStrategy
from .recipes import generate_recipes

logger = logging.getLogger(__name__)


class SystematicStrategy(DifficultyStrategy):
    """
    Systematic strategy: uses a fixed difficulty level.
    """
    
    def __init__(self, difficulty: int):
        """
        Initializes the systematic strategy.
        
        Args:
            difficulty (int): Fixed difficulty level.
        """
        # CORREÇÃO: Chamada ao super init
        super().__init__(difficulty=difficulty)
        
        self.difficulty = difficulty
        all_recipes = generate_recipes(difficulty)
        self.recipe = all_recipes.get(difficulty)
        
        if not self.recipe:
            raise ValueError(f"Invalid difficulty: {difficulty}")
        
        logger.info(f"SystematicStrategy initialized: difficulty={difficulty}")
    
    def get_strategy_name(self) -> str:
        return "systematic"
    
    def select_difficulty(self, **context) -> Tuple[int, Dict]:
        """Returns the fixed difficulty and recipe."""
        return self.difficulty, self.recipe
    
    def get_fallback_strategy(self) -> Optional['SystematicStrategy']:
        """Creates a fallback strategy with reduced difficulty."""
        if self.difficulty <= 1:
            logger.warning("Already at minimum difficulty, no fallback available.")
            return None
        
        fallback_difficulty = self.difficulty - 1
        logger.info(f"Creating fallback strategy: difficulty={fallback_difficulty}")
        
        return SystematicStrategy(fallback_difficulty)
