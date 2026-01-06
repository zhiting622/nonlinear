"""
Piece utilities for data segmentation
"""

import numpy as np


def get_piece_starting_indices(y, piece_len: int, overlap_ratio: float):
    """
    Generate array of starting indices for each piece of y.
    
    Args:
        y: input array
        piece_len: length of each piece
        overlap_ratio: ratio of overlap between pieces
        
    Returns:
        List of starting indices for each piece
    """
    if piece_len <= 0:
        raise ValueError("piece_len must be a positive integer")
    
    n = len(y)
    step = piece_len * (1 - overlap_ratio)
    
    # Only windows that fully fit within y
    starting_indices = []
    for i in range(0, n - piece_len + 1, int(step)):
        starting_indices.append(i)
    
    return starting_indices



def get_num_pieces(y, piece_len: int, overlap_ratio: float = 0.0):
    """
    Get the total number of pieces that can be extracted from y.
    
    Args:
        y: input array
        piece_len: length of each piece
        overlap_ratio: ratio of overlap between pieces
        
    Returns:
        Number of pieces
    """
    starting_indices = get_piece_starting_indices(y, piece_len, overlap_ratio)
    return len(starting_indices)
