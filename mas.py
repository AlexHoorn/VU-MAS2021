import random
from functools import lru_cache
from math import log, sqrt
from time import perf_counter
from typing import Dict, List, Tuple

from numpy import e


def edit_distance(X: Tuple, Y: Tuple) -> int:
    # Edit distance
    d = 0
    for x, y in zip(X, Y):
        if x != y:
            d += 1

    return d


class BinaryTree:
    def __init__(self, depth: int, choices: List = ["L", "R"]) -> None:
        # BinaryTree that generates rewards lazily so they can be of galactic size

        self.depth = int(depth)  # depth of the tree
        self.choices = list(choices)  # the choices at each node

        self._target = self._generate_target()  # random path to base rewards on
        self.n_paths = len(choices) ** depth  # amount of paths in the tree

    # Cache last million
    @lru_cache(maxsize=1000000)
    def get_reward(self, path: Tuple) -> float:
        # Get a reward given a path
        return self._reward_function(self._target, path)

    def _generate_target(self):
        # Generate random path
        return tuple(random.choice(self.choices) for _ in range(self.depth))

    def _reward_function(
        self, target: Tuple, path: Tuple, B: float = 2.0, r: float = 20.0
    ):
        return B * e ** -(edit_distance(target, path) / r)


class MCTS:
    def __init__(self, tree: BinaryTree, c, iterations=5, verbose=False) -> None:
        self.tree = tree  # the binary tree
        self.c = c  # c for ucb
        self.iterations = iterations  # allowed iterations for rollout
        self.verbose = verbose  # print at end of run

    def run(self):
        # Keep track of metrics
        self.search_steps = 0
        runtime = perf_counter()

        # Dictionaries to track rewards and n visits
        t, n = {(): 0}, {(): 0}

        while True:
            # Select current best path
            path = self._select_path(t, n)
            self.search_steps += 1

            # Break if we found the leaf
            if len(path) == self.tree.depth:
                break

            # Determine rewards for child nodes
            for c in self.tree.choices:
                curr_path = path + tuple(c)

                t[curr_path] = self._roll_out(curr_path)
                n[curr_path] = 1

                # Back the values up
                self._backup(curr_path, t)
                self._backup(curr_path, n)

        # Determine runtime
        runtime = perf_counter() - runtime

        # Store attributes after run
        self.t = t
        self.n = n
        self.best = path
        self.reward = self.tree.get_reward(path)
        self.runtime = runtime

        # Print after run
        if self.verbose:
            reward = self.reward
            print(
                f"Found {reward=:.2f} with path=`{''.join(path)}` in {runtime:.2f}s and {self.search_steps} steps"
            )

    def _backup(self, path: Tuple, d: Dict[Tuple, float]) -> Dict[Tuple, float]:
        # Backup along the path
        addition = d[path]
        for i in range(len(path)):
            d[path[:i]] += addition

    def _select_path(self, t: Dict[Tuple, float], n: Dict[Tuple, float]) -> Tuple:
        # Select the current best path
        path = tuple()
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
            for _ in range(self.iterations)
        ]

        return max(rewards)

    def _random_path(self, depth: int) -> Tuple:
        # Generate a random path
        return tuple(random.choice(self.tree.choices) for _ in range(depth))

    def _ucb(self, node_value: float, visits_parent: int, visits_node: int) -> float:
        # Calculate UCB
        return node_value + self.c * sqrt(log(visits_parent) / visits_node)
