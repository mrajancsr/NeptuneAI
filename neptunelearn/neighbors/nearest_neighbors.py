# pyre-strict
from __future__ import annotations

import heapq
import sys
from dataclasses import dataclass
from typing import List, Tuple

from numpy.linalg import norm
from numpy.typing import NDArray


@dataclass
class kNeighborsClassifier:
    n_neighbors: int
    metric: str = "eucledean"

    def _1NN(
        self, examples: List[NDArray], targets: NDArray, query: NDArray
    ) -> List[Tuple[NDArray, NDArray]]:
        if self.n_neighbors != 1:
            raise ValueError("Number of neighbors requested is not equal to 1")
        closest_point = None
        min_dist = sys.maxsize
        curr_dist = None

        for index, x in enumerate(examples):
            curr_dist = norm(x - query)
            if curr_dist < min_dist:
                closest_point = (x, targets[index])
                min_dist = curr_dist
        return closest_point

    def nearest_neighbors(
        self, examples: List[NDArray], targets: NDArray, query: NDArray
    ) -> List[Tuple[NDArray, NDArray]]:
        if self.n_neighbors == 1:
            return self._1NN(examples, targets, query)
        heap = []
        for xi, yi in zip(examples, targets):
            dist = norm(xi - query)
            if len(heap) == self.n_neighbors:
                heapq.heappop(heap)
            else:
                heapq.heappush(heap, (dist, xi, yi))
        assert len(heap) == self.n_neighbors, "not equal neighbors"
        return [(x, y) for (dist, x, y) in heap]
