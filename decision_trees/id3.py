# pyre-strict
"""Implementation of ID3(Iterative Dichotomiser 3) Algorithm
based on Mitchell (1997)
Author: Rajan Subramanian
Date: 
Notes:
    - This Algorithm is only applicable for classification tasks
    - todo Also include the c4.5 algorithm of Quinian (rule post pruning)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ID3:
    criterion: str = ("entropy",)
    max_depth: Optional[int] = None
    prune: bool = False


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


import pandas as pd

df = pd.DataFrame(
    [
        ["green", "M", 10.1, "class1"],
        ["red", "L", 13.5, "class2"],
        ["blue", "XL", 15.3, "class1"],
    ]
)

X = df.to_numpy()

print(X)
