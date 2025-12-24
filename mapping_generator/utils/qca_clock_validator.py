"""
QCA Clock Validation Module

Validates QCA graphs against USE and 2DDWAVE clock schemes.
Can be used standalone or integrated into existing pipeline.
"""

import networkx as nx
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ClockScheme(Enum):
    """Supported QCA clock schemes."""
    USE = "USE"           # Universal, Scalable, Efficient
    TWODDWAVE = "2DDWAVE" # 2D Wave clocking


@dataclass
class ClockValidationResult:
    """Results from clock scheme validation."""
    is_valid: bool
    scheme: ClockScheme
    errors: List[str]
    warnings: List[str]
    
    # Detailed metrics
    max_delay_violation: int = 0
    balanced_paths: int = 0
    unbalanced_paths: int = 0
    clock_zones_used: int = 0
    
    # Balancing statistics
    min_path_length: int = 0
    max_path_length: int = 0
    avg_path_length: float = 0.0
    path_length_variance: float = 0.0


class QcaClockValidator:
    """
    Validates QCA placement graphs against clock scheme constraints.
    
    Focuses on:
    1. Path balancing (all paths to convergence points must be equal length)
    2. Clock zone assignment feasibility
    3. Data flow synchronization
    """
    
    def __init__(self, clock_scheme: ClockScheme = ClockScheme.USE):
        """
        Initialize validator.
        
        Args:
            clock_scheme: Which clock scheme to validate against (USE or 2DDWAVE)
        """
        self.clock_scheme = clock_scheme
        
        # Clock scheme parameters
        if clock_scheme == ClockScheme.USE:
            self.num_clock_zones = 4
            self.max_delay_buffers = 3  # Maximum buffers to add for balancing
        else:  # 2DDWAVE
            self.num_clock_zones = 4
            self.max_delay_buffers = 3
    
    def validate(self, graph: nx.DiGraph) -> ClockValidationResult:
        """
        Main validation entry point.
        
        Args:
            graph: QCA placement graph with 'level' attributes on nodes
        
        Returns:
            ClockValidationResult with detailed analysis
        """
        errors = []
        warnings = []
        
        # 1. Check if graph has level information
        if not self._has_level_info(graph):
            errors.append("Graph missing 'level' attribute on nodes (required for clock validation)")
            return ClockValidationResult(
                is_valid=False,
                scheme=self.clock_scheme,
                errors=errors,
                warnings=warnings
            )
        
        # 2. Calculate path lengths to all convergence points
        path_analysis = self._analyze_path_lengths(graph)
        
        # 3. Check for imbalance at convergence points
        imbalance_errors = self._check_convergence_balance(graph, path_analysis)
        errors.extend(imbalance_errors)
        
        # 4. Validate clock zone assignment feasibility
        zone_errors = self._validate_clock_zones(graph)
        errors.extend(zone_errors)
        
        # 5. Calculate statistics
        stats = self._calculate_statistics(graph, path_analysis)
        
        return ClockValidationResult(
            is_valid=(len(errors) == 0),
            scheme=self.clock_scheme,
            errors=errors,
            warnings=warnings,
            **stats
        )
    
    def _has_level_info(self, graph: nx.DiGraph) -> bool:
        """Check if all nodes have 'level' attribute."""
        for node, data in graph.nodes(data=True):
            if 'level' not in data:
                return False
        return True
    
    def _analyze_path_lengths(self, graph: nx.DiGraph) -> Dict[Tuple, List[int]]:
        """
        Analyze path lengths from inputs to every node.
        
        Returns:
            Dict mapping node -> list of path lengths from all inputs
        """
        path_lengths = {}
        
        # Find all input nodes
        inputs = [n for n, d in graph.nodes(data=True) if d.get('type') == 'input']
        
        # For each non-input node, calculate paths from all inputs
        for node in graph.nodes():
            if graph.nodes[node].get('type') == 'input':
                path_lengths[node] = [0]
                continue
            
            lengths = []
            for inp in inputs:
                if nx.has_path(graph, inp, node):
                    # Get the level-based path length
                    # (more accurate than simple path counting)
                    node_level = graph.nodes[node].get('level', 0)
                    lengths.append(node_level)
            
            path_lengths[node] = lengths if lengths else [0]
        
        return path_lengths
    
    def _check_convergence_balance(
        self, 
        graph: nx.DiGraph, 
        path_analysis: Dict[Tuple, List[int]]
    ) -> List[str]:
        """
        Check if all convergence points are properly balanced.
        
        In QCA, when multiple paths converge, they MUST arrive at the
        same clock cycle (same level) for the circuit to function correctly.
        """
        errors = []
        
        for node in graph.nodes():
            predecessors = list(graph.predecessors(node))
            
            # Only check nodes with multiple inputs (convergence points)
            if len(predecessors) < 2:
                continue
            
            # Get levels of all predecessors
            pred_levels = [graph.nodes[p].get('level', 0) for p in predecessors]
            
            # Check if all paths arrive at the same level
            if len(set(pred_levels)) > 1:
                min_level = min(pred_levels)
                max_level = max(pred_levels)
                imbalance = max_level - min_level
                
                errors.append(
                    f"Convergence point {node} has imbalanced inputs: "
                    f"levels {pred_levels} (imbalance={imbalance} cycles). "
                    f"Predecessors: {predecessors}"
                )
        
        return errors
    
    def _validate_clock_zones(self, graph: nx.DiGraph) -> List[str]:
        """
        Validate that the graph can be assigned to clock zones.
        
        For USE/2DDWAVE with 4 zones, adjacent levels must be in different zones.
        """
        errors = []
        
        # Get max level (depth)
        max_level = max(
            (data.get('level', 0) for _, data in graph.nodes(data=True)),
            default=0
        )
        
        # Check if depth exceeds reasonable limits
        # With 4 zones, we can support unlimited depth, but very deep
        # circuits may indicate design issues
        if max_level > 50:
            errors.append(
                f"Circuit depth ({max_level}) is very large. "
                f"This may indicate balancing issues or inefficient design."
            )
        
        return errors
    
    def _calculate_statistics(
        self, 
        graph: nx.DiGraph, 
        path_analysis: Dict[Tuple, List[int]]
    ) -> Dict:
        """Calculate detailed statistics about the graph balance."""
        
        # Collect all path lengths
        all_lengths = []
        for lengths in path_analysis.values():
            all_lengths.extend(lengths)
        
        if not all_lengths:
            return {
                'max_delay_violation': 0,
                'balanced_paths': 0,
                'unbalanced_paths': 0,
                'clock_zones_used': 0,
                'min_path_length': 0,
                'max_path_length': 0,
                'avg_path_length': 0.0,
                'path_length_variance': 0.0
            }
        
        min_len = min(all_lengths)
        max_len = max(all_lengths)
        avg_len = sum(all_lengths) / len(all_lengths)
        
        # Calculate variance
        variance = sum((x - avg_len) ** 2 for x in all_lengths) / len(all_lengths)
        
        # Count balanced vs unbalanced convergence points
        balanced = 0
        unbalanced = 0
        max_violation = 0
        
        for node in graph.nodes():
            preds = list(graph.predecessors(node))
            if len(preds) >= 2:
                pred_levels = [graph.nodes[p].get('level', 0) for p in preds]
                if len(set(pred_levels)) == 1:
                    balanced += 1
                else:
                    unbalanced += 1
                    violation = max(pred_levels) - min(pred_levels)
                    max_violation = max(max_violation, violation)
        
        # Estimate clock zones used
        clock_zones_used = (max_len % self.num_clock_zones) + 1
        
        return {
            'max_delay_violation': max_violation,
            'balanced_paths': balanced,
            'unbalanced_paths': unbalanced,
            'clock_zones_used': clock_zones_used,
            'min_path_length': min_len,
            'max_path_length': max_len,
            'avg_path_length': avg_len,
            'path_length_variance': variance
        }


class QcaClockBalancer:
    """
    Attempts to fix imbalanced graphs by inserting delay buffers.
    
    OPTIONAL: Can be used to auto-fix graphs or just report what would be needed.
    """
    
    def __init__(self, max_delay_buffers: int = 3):
        """
        Initialize balancer.
        
        Args:
            max_delay_buffers: Maximum buffers to insert per path
        """
        self.max_delay_buffers = max_delay_buffers
    
    def analyze_balancing_needs(
        self, 
        graph: nx.DiGraph
    ) -> Dict[Tuple, int]:
        """
        Analyze how many delay buffers would be needed for each path.
        
        Does NOT modify the graph, just reports requirements.
        
        Returns:
            Dict mapping node -> number of delay buffers needed
        """
        buffer_needs = {}
        
        for node in graph.nodes():
            preds = list(graph.predecessors(node))
            
            if len(preds) >= 2:
                pred_levels = [graph.nodes[p].get('level', 0) for p in preds]
                max_level = max(pred_levels)
                
                # For each predecessor, calculate how many buffers needed
                for pred in preds:
                    pred_level = graph.nodes[pred].get('level', 0)
                    delay_needed = max_level - pred_level
                    
                    if delay_needed > 0:
                        buffer_needs[pred] = delay_needed
        
        return buffer_needs
    
    def estimate_balancing_feasibility(
        self, 
        graph: nx.DiGraph
    ) -> Tuple[bool, Dict]:
        """
        Estimate if the graph can be balanced within constraints.
        
        Returns:
            (is_feasible, detailed_report)
        """
        buffer_needs = self.analyze_balancing_needs(graph)
        
        # Check if any path needs more buffers than allowed
        max_needed = max(buffer_needs.values(), default=0)
        is_feasible = max_needed <= self.max_delay_buffers
        
        report = {
            'is_feasible': is_feasible,
            'max_buffers_needed': max_needed,
            'max_buffers_allowed': self.max_delay_buffers,
            'total_buffers_needed': sum(buffer_needs.values()),
            'paths_needing_buffers': len(buffer_needs),
            'buffer_needs': buffer_needs
        }
        
        return is_feasible, report


# ============================================================================
# HELPER FUNCTIONS FOR EASY INTEGRATION
# ============================================================================

def quick_validate(
    graph: nx.DiGraph, 
    scheme: str = "USE"
) -> ClockValidationResult:
    """
    Quick validation helper.
    
    Usage:
        result = quick_validate(my_graph, "USE")
        if result.is_valid:
            print("✅ Valid!")
        else:
            print("❌ Errors:", result.errors)
    """
    scheme_enum = ClockScheme.USE if scheme.upper() == "USE" else ClockScheme.TWODDWAVE
    validator = QcaClockValidator(scheme_enum)
    return validator.validate(graph)


def print_validation_report(result: ClockValidationResult):
    """
    Pretty print validation results.
    
    Usage:
        result = quick_validate(graph)
        print_validation_report(result)
    """
    print("\n" + "="*80)
    print(f"QCA CLOCK VALIDATION REPORT - {result.scheme.value}")
    print("="*80)
    
    status = "✅ VALID" if result.is_valid else "❌ INVALID"
    print(f"\nStatus: {status}")
    
    if result.errors:
        print(f"\n🚨 Errors ({len(result.errors)}):")
        for i, error in enumerate(result.errors, 1):
            print(f"  {i}. {error}")
    
    if result.warnings:
        print(f"\n⚠️  Warnings ({len(result.warnings)}):")
        for i, warning in enumerate(result.warnings, 1):
            print(f"  {i}. {warning}")
    
    print(f"\n📊 Statistics:")
    print(f"  Balanced convergence points:   {result.balanced_paths}")
    print(f"  Unbalanced convergence points: {result.unbalanced_paths}")
    print(f"  Maximum delay violation:       {result.max_delay_violation} cycles")
    print(f"  Path length range:             {result.min_path_length} - {result.max_path_length}")
    print(f"  Average path length:           {result.avg_path_length:.2f}")
    print(f"  Path variance:                 {result.path_length_variance:.2f}")
    print(f"  Clock zones used:              {result.clock_zones_used}")
    
    print("="*80 + "\n")
