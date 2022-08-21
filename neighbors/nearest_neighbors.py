# pyre-strict
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import List, Tuple

from numpy.linalg import norm
from numpy.typing import NDArray


@dataclass
class kNeighborsClassifier:
    n_neighbors: int
    metric: str = "eucledean"

    def nearest_neighbors(
        self, examples: List[NDArray], targets: NDArray, query: NDArray
    ) -> List[Tuple[NDArray, NDArray]]:
        heap = []
        for xi, yi in zip(examples, targets):
            dist = norm(xi - query)
            if len(heap) == self.n_neighbors:
                heapq.heappop(heap)
            else:
                heapq.heappush(heap, (dist, xi, yi))
        assert len(heap) == self.n_neighbors, "not equal neighbors"
        return [(x, y) for (dist, x, y) in heap]
