# mapping_generator/generation/strategies/recipes.py

"""
Difficulty recipe generator.
Centralized to avoid circular imports.
"""

from typing import Dict

def generate_recipes(max_difficulty: int) -> Dict[int, Dict[str, int]]:
    """
    Generates recipes with increasing difficulty levels.
    
    Args:
        max_difficulty (int): Maximum difficulty level (e.g., 20).
    
    Returns:
        Dict: A mapping of {difficulty_level: {'reconvergence': r, 'convergence': c}}.
    """
    recipes = {1: {"reconvergence": 0, "convergence": 0}}
    
    if max_difficulty <= 1:
        return recipes
    
    r, c = 0, 0
    for i in range(2, max_difficulty + 1):
        if r == c:
            c += 1
        elif c > r:
            r, c = c, r
        elif r > c + 1:
            c += 1
        else:
            r, c = c, r + 1
        
        recipes[i] = {"reconvergence": r, "convergence": c}
    
    return recipes
