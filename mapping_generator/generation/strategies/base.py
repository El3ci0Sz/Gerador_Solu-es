# mapping_generator/generation/strategies/base.py

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


class DifficultyStrategy(ABC):
    """
    Base interface for difficulty selection strategies.
    """
    
    def __init__(self, **config):
        """
        Initializes the strategy with specific configuration.
        
        Args:
            **config: Strategy-specific parameters stored for statistics/logging.
        """
        self.config = config
    
    @abstractmethod
    def select_difficulty(self, **context) -> Tuple[Union[int, str], Optional[Dict]]:
        """
        Selects the difficulty level and recipe for the next generation attempt.
        
        Args:
            **context: Contextual data needed for decision (e.g., graph_size, k_range).
        
        Returns:
            Tuple containing:
            - difficulty: int (level) or str (identifier).
            - recipe: dict with generation parameters (e.g. {'reconvergence': 2}) or None.
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Returns the unique identifier name of the strategy."""
        pass
    
    def on_success(self, difficulty: Union[int, str]):
        """Hook called when generation with this difficulty succeeds."""
        pass
    
    def on_failure(self, difficulty: Union[int, str]):
        """Hook called when generation with this difficulty fails."""
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Returns internal statistics of the strategy."""
        return {
            'strategy': self.get_strategy_name(),
            'config': self.config
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.config})"
