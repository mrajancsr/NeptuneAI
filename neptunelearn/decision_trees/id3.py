# pyre-strict
"""Implementation of ID3(Iterative Dichotomiser 3) Algorithm
based on Mitchell (1997)
Author: Rajan Subramanian
Date:
Notes:
    - This Algorithm is only applicable for classification tasks
    - todo Also include the c4.5 algorithm of Quinian (rule post pruning)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np


@dataclass
class Tree:
    data: Any
    left: Optional[Tree] = None
    right: Optional[Tree] = None


@dataclass
class ID3:
    criterion: str = ("entropy",)
    max_depth: Optional[int] = None
    prune: bool = False

    def grow_tree(self, examples: np.ndarray, features: List[str]) -> Tree:
        """Grows a tree based on examples D

        Parameters
        ----------
        examples : np.ndarray
            data D contains features and labels
        features : List[str]
            name of each feature on the examples D

        Returns
        -------
        Tree
            Fully Grown decision Tree that classifies examples D
        """
        pass


def assign_label(examples: np.ndarray):
    """Assigns majority label to examples
    Assumes last column contains the labels

    Parameters
    ----------
    examples : np.ndarray,
                Each row is observation
        training examples
    """
    labels = examples[:, -1]
    majority_labels = {}
    for label in labels:
        majority_labels[label] = majority_labels.get(label, 0) + 1
    return max(majority_labels, key=majority_labels.get)


def homogenous(examples: np.ndarray) -> bool:
    """Returns True of examples are homogenous, False otherwise

    Parameters
    ----------
    examples : np.ndarray
        _description_

    Returns
    -------
    bool
        _description_
    """
    labels = examples[:, -1]
    sample_length = len(labels)
    majority_labels = {}
    for label in labels:
        majority_labels[label] = majority_labels.get(label, 0) + 1
    for label in majority_labels:
        if majority_labels[label] == sample_length:
            return True
    return False


def best_split(examples: np.ndarray, features: List[str]) -> str:
    """Returns the best feature to split on

    Parameters
    ----------
    examples : np.ndarray
        _description_
    features : List[str]
        _description_

    Returns
    -------
    str
        _description_
    """
    pass
