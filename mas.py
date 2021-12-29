import random
from functools import cache
from math import log, sqrt
from time import perf_counter
from typing import Dict, List, Tuple

from numpy import e


def edit_distance(X: Tuple, Y: Tuple) -> int:
    assert len(X) == len(Y), "X and Y must match in length."
    # Edit distance
    d = 0
    for x, y in zip(X, Y):
        if x != y:
            d += 1

    return d


class BinaryTree:
    def __init__(self, depth: int, choices: List = ["L", "R"]) -> None:
        # BinaryTree that can be of galactic size because it generates rewards lazily

        self.depth = int(depth)  # depth of the tree
        self.choices = list(choices)  # the choices at each node

        self.target = self._generate_target()  # random path to base rewards on
        self.n_paths = len(choices) ** depth  # amount of paths in the tree

    @cache
    def get_reward(self, path: Tuple) -> float:
        # Get a reward given a path
        return self._reward_function(self.target, path)

    def _generate_target(self):
        # Generate random path
        return tuple(random.choice(self.choices) for _ in range(self.depth))

    def _reward_function(self, target: Tuple, path: Tuple, B: float = 2, r: float = 20):
        return B * e ** -(edit_distance(target, path) / r)


class MCTS:
    def __init__(
        self,
        tree: BinaryTree,
        c,
        iterations=50,
        max_patience=5,
        rollouts=5,
    ) -> None:
        self.tree = tree  # the binary tree
        self.c = c  # c for ucb function
        self.iterations = iterations  # iterations per setting of root
        self.max_patience = max_patience  # so we can break iterations early
        self.rollouts = rollouts  # allowed iterations for rollout

    def run(self):
        path = tuple()  # new empty path
        # Dictionaries to track rewards and n visits
        t, n = {path: 0}, {path: 0}

        # Keep track of metrics
        self.search_iterations = 0
        self.early_stops = 0
        runtime = perf_counter()

        # Construct path by iteratively setting new roots
        for _ in range(self.tree.depth):
            path += tuple(self._determine_root(path, t, n))

        # Determine runtime
        runtime = perf_counter() - runtime

        # Store attributes after run
        self.path = path
        self.reward = self.tree.get_reward(path)
        self.runtime = runtime
        self.t = t
        self.n = n

    def print_solution(self):
        reward = self.reward
        path = self.path
        runtime = self.runtime
        steps = self.search_iterations

        print(
            f"Found {reward=:.2f} with path=`{''.join(path)}` in {runtime:.2f}s and {steps} steps"
        )

    def _determine_root(self, path: Tuple, t, n):
        # Keep note of patience
        patience = 0
        # Iterate to max iterations
        for _ in range(self.iterations):

            # Select current path by ucb
            curr_path = self._select_path(path, t, n)
            self.search_iterations += 1

            # This implements early breaking, if we keep getting root nodes then stop iterating
            if len(curr_path) >= self.tree.depth:
                patience += 1
                # Break if we keep selecting leaf nodes
                if patience >= self.max_patience:
                    self.early_stops += 1
                    break

            else:
                # Create and explore child nodes. This is the biggest difference to the MCTS in the slides
                # when we reach an unexplored area this immediately expands the DIRECT children. Instead of
                # setting t = 0 and n = 0. This is much easier to do programmatically.
                for c in self.tree.choices:
                    child = curr_path + tuple(c)
                    # Determine reward by rollout
                    t[child] = self._roll_out(child)
                    n[child] = 1
                    # Back the values up
                    self._backup(child, t)
                    self._backup(child, n)

        return self._select_choice(path, t, n)

    def _backup(self, path: Tuple, d: Dict[Tuple, float]) -> Dict[Tuple, float]:
        # Backup along the path
        addition = d[path]
        for i in range(len(path)):
            d[path[:i]] += addition

    def _select_path(
        self, path: Tuple, t: Dict[Tuple, float], n: Dict[Tuple, float]
    ) -> Tuple:
        # Select the current best path
        while True:
            # Make choices until we reach an end
            try:
                path += self._select_choice(path, t, n)
            except KeyError:
                break

        return path

    def _select_choice(self, path, t, n):
        ucbs = {}

        # For each choice we can make from the node
        for c in self.tree.choices:
            c = tuple(c)
            curr_path = path + c

            # Determine UCB
            ucbs[c] = self._ucb(t[curr_path], n[path], n[curr_path])

        # Return the best choice
        return max(ucbs, key=lambda x: ucbs[x])

    def _roll_out(self, path: Tuple) -> float:
        # Roll out by randomly generating path and fetching their rewards
        random_depth = self.tree.depth - len(path)
        rewards = [
            self.tree.get_reward(path + self._random_path(random_depth))
            for _ in range(self.rollouts)
        ]

        return max(rewards)

    def _random_path(self, depth: int) -> Tuple:
        # Generate a random path
        return tuple(random.choice(self.tree.choices) for _ in range(depth))

    def _ucb(self, node_value: float, visits_parent: int, visits_node: int) -> float:
        # Calculate UCB
        return node_value + self.c * sqrt(log(visits_parent) / visits_node)
