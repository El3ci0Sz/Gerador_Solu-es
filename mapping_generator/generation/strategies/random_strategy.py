# mapping_generator/generation/strategies/random_strategy.py

import random
import logging
from typing import Tuple, Dict, Any, Optional, Union
from collections import defaultdict
from .base import DifficultyStrategy
from .recipes import generate_recipes

logger = logging.getLogger(__name__)


class RandomStrategy(DifficultyStrategy):
    """
    Random and adaptive difficulty strategy.
    """
    
    def __init__(self, difficulty_range: Tuple[int, int], adaptive: bool = False):
        """
        Initializes the random strategy.
        
        Args:
            difficulty_range (Tuple[int, int]): Min and max difficulty.
            adaptive (bool): If True, adjusts weights based on success rate.
        """
        # CORREÇÃO: Chamada ao super init para salvar config
        super().__init__(difficulty_range=difficulty_range, adaptive=adaptive)
        
        self.difficulty_range = difficulty_range
        self.adaptive = adaptive
        
        min_diff, max_diff = difficulty_range
        if min_diff < 1 or max_diff < min_diff:
            raise ValueError(f"Invalid range: {difficulty_range}")
        
        self.recipes = generate_recipes(max_diff)
        
        # Validate range availability
        for d in range(min_diff, max_diff + 1):
            if d not in self.recipes:
                raise ValueError(f"Difficulty {d} outside supported range.")
        
        # Statistics for adaptive mode
        self._success_by_difficulty = defaultdict(int)
        self._failure_by_difficulty = defaultdict(int)
        self._weights = self._initialize_weights()
    
    def _initialize_weights(self) -> Dict[int, float]:
        """Initializes uniform weights for each difficulty."""
        min_diff, max_diff = self.difficulty_range
        return {d: 1.0 for d in range(min_diff, max_diff + 1)}
    
    def select_difficulty(self, **context) -> Tuple[int, Dict[str, int]]:
        """
        Selects a random difficulty (weighted if adaptive).
        
        Args:
            **context: Must contain 'graph_size' and 'k_range'.
        """
        # CORREÇÃO: Extrair argumentos do context para respeitar assinatura da Base
        graph_size = context.get('graph_size')
        k_range = context.get('k_range')
        
        if graph_size is None or k_range is None:
            # Fallback seguro se o contexto não for passado
            # (embora o gerador deva sempre passar)
            logger.warning("RandomStrategy missing context (graph_size/k_range). Using pure random.")
            d = random.randint(self.difficulty_range[0], self.difficulty_range[1])
            return d, self.recipes[d]

        min_diff, max_diff = self.difficulty_range
        k_min = k_range[0]
        
        # Filter viable difficulties based on graph size constraints
        viable_difficulties = []
        for d in range(min_diff, max_diff + 1):
            recipe = self.recipes[d]
            r = recipe.get('reconvergence', 0)
            c = recipe.get('convergence', 0)
            
            # Minimum node cost calculation
            min_nodes_needed = 1 + (r * (k_min + 1)) + (c * k_min)
            
            if graph_size >= min_nodes_needed:
                viable_difficulties.append(d)
        
        if not viable_difficulties:
            logger.debug(f"No viable difficulty for size={graph_size}. Fallback to difficulty=1.")
            return 1, self.recipes[1]
        
        if self.adaptive:
            weights = [self._weights[d] for d in viable_difficulties]
            difficulty = random.choices(viable_difficulties, weights=weights, k=1)[0]
        else:
            difficulty = random.choice(viable_difficulties)
        
        return difficulty, self.recipes[difficulty]
    
    def on_success(self, difficulty: Union[int, str]):
        """Increments success count and updates weights if adaptive."""
        if isinstance(difficulty, int):
            self._success_by_difficulty[difficulty] += 1
            if self.adaptive:
                self._update_weights()
    
    def on_failure(self, difficulty: Union[int, str]):
        """Increments failure count and updates weights if adaptive."""
        if isinstance(difficulty, int):
            self._failure_by_difficulty[difficulty] += 1
            if self.adaptive:
                self._update_weights()
    
    def _update_weights(self):
        """Updates selection weights based on success rate."""
        min_diff, max_diff = self.difficulty_range
        
        for d in range(min_diff, max_diff + 1):
            total = self._success_by_difficulty[d] + self._failure_by_difficulty[d]
            
            if total < 10:
                continue
            
            success_rate = self._success_by_difficulty[d] / total
            self._weights[d] = max(0.1, success_rate * 2.0)
        
        # Normalize
        total_weight = sum(self._weights.values())
        if total_weight > 0:
            for d in self._weights:
                self._weights[d] /= total_weight
    
    def get_strategy_name(self) -> str:
        min_diff, max_diff = self.difficulty_range
        adaptive_suffix = "_adaptive" if self.adaptive else ""
        return f"random_{min_diff}-{max_diff}{adaptive_suffix}"
    
    def get_fallback_strategy(self) -> Optional[DifficultyStrategy]:
        return None
