from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Any, Tuple
import logging

if TYPE_CHECKING:
    from ..QcaGrammarGenerator import QcaGrammarGenerator

logger = logging.getLogger(__name__)


class BaseGrammarRule(ABC):
    """
    Base interface for all grammar rules.
    
    Defines the contract that all rules must follow, ensuring:
    - Extensibility: Easy addition of new rules.
    - Testability: Clear interface for unit tests.
    - Consistency: All rules behave similarly.
    """
    
    def __init__(self, **config):
        """
        Initializes the rule with specific configurations.
        
        Args:
            **config: Rule-specific parameters (e.g., k_range, max_path_length).
        """
        self.config = config
        self._application_count = 0
    
    @abstractmethod
    def apply(self, generator: 'QcaGrammarGenerator', start_node: Tuple[int, int]) -> bool:
        """
        Applies the grammar rule starting from a specific node.
        
        This method MUST modify the generator's graph if successful.
        
        Args:
            generator: The main generator instance.
            start_node: Coordinate (row, col) where the rule starts.
            
        Returns:
            True if applied successfully (graph modified).
            False if application failed (graph unchanged).
        """
        pass
    
    @abstractmethod
    def can_apply(self, generator: 'QcaGrammarGenerator', start_node: Tuple[int, int]) -> bool:
        """
        Checks if the rule CAN be applied without modifying the graph.
        
        Args:
            generator: The main generator instance.
            start_node: Candidate node.
            
        Returns:
            True if the rule has potential to be applied.
            False if it definitely cannot be applied.
        """
        pass
    
    @abstractmethod
    def get_rule_type(self) -> str:
        """Returns the rule identifier (e.g., 'tree', 'reconvergence')."""
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Returns rule usage statistics."""
        return {
            'rule_type': self.get_rule_type(),
            'application_count': self._application_count,
            'config': self.config
        }
    
    def _increment_counter(self):
        """Increments application counter."""
        self._application_count += 1
    
    def reset_statistics(self):
        """Resets rule statistics."""
        self._application_count = 0
    
    def __repr__(self) -> str:
        config_str = ', '.join(f"{k}={v}" for k, v in self.config.items())
        return f"{self.__class__.__name__}({config_str})"
    
    def __str__(self) -> str:
        return f"{self.get_rule_type()} rule (applied {self._application_count} times)"
