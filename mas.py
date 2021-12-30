import random
from functools import cache
from math import log, sqrt
from time import perf_counter
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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
        return B * np.e ** -(edit_distance(target, path) / r)


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
                # setting t = 0 and n = 0. This is much easier to do programmatically instead of dealing with
                # unexplored children.
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
        depth = self.tree.depth - len(path)
        rewards = [
            self.tree.get_reward(path + self._random_path(depth))
            for _ in range(self.rollouts)
        ]

        return max(rewards)

    def _random_path(self, depth: int) -> Tuple:
        # Generate a random path
        return tuple(random.choice(self.tree.choices) for _ in range(depth))

    def _ucb(self, node_value: float, visits_parent: int, visits_node: int) -> float:
        # Calculate UCB
        return node_value + self.c * sqrt(log(visits_parent) / visits_node)


class GridWorld:
    def __init__(
        self,
        terminals: Dict[Tuple, float],
        walls: List[Tuple] = [],
        gridsize: Tuple[int, int] = (9, 9),
        step_cost: float = -1.0,
    ) -> None:
        # Construct grid with rewards
        grid = np.ones(gridsize) * step_cost
        for pos, reward in terminals.items():
            grid[pos] = reward

        # Construct grid with walls
        grid_wall = np.zeros(gridsize).astype(bool)
        for wall in walls:
            grid_wall[wall] = True

        self.terminals = list(terminals.keys())
        self.walls = walls
        self.step_cost = step_cost
        self.gridsize = gridsize

        self.grid = grid
        self.grid_wall = grid_wall
        # The possible actions that can be taken
        self.actions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        # The possible starting positions (including terminal)
        self.start_positions = list(zip(*np.where(~self.grid_wall)))

    def plot_grid(self):
        # Plot the grid and its rewards
        _, ax = plt.subplots(figsize=(4, 4))

        ax = sns.heatmap(self.grid, cmap="RdYlGn", annot=True, cbar=False, ax=ax)
        self.plot_walls(ax=ax)

        return ax

    def plot_walls(self, ax=None):
        # Plot the walls in the grid
        ax = sns.heatmap(
            self.grid_wall,
            mask=self.grid_wall != True,
            cmap="Blues",
            vmin=0,
            vmax=1.1,
            cbar=False,
            ax=ax,
        )

        return ax

    def step(
        self, pos: Tuple[int, int], step: Tuple[int, int]
    ) -> Tuple[Tuple, float, bool]:
        # The new position after the step
        new_pos = tuple(x + y for x, y in zip(pos, step))

        # Make sure we stay in the grid or bounce back if we hit a wall
        if not self._pos_in_world(new_pos) or self._pos_in_wall(new_pos):
            new_pos = pos
            # Incur cost of making a step
            reward = self.step_cost
        else:
            # Reward from grid
            reward = self.grid[new_pos]

        # New position, reward, terminal state
        return new_pos, reward, new_pos in self.terminals

    def _pos_in_world(self, pos: Tuple[int, int]):
        # Check if given position is within the bounds of the grid
        y, x = pos
        yg, xg = self.gridsize
        return xg > x >= 0 and yg > y >= 0

    def _pos_in_wall(self, pos):
        # Check if given position is in a wall
        return pos in self.walls


class MonteCarloSweep:
    def __init__(self, world: GridWorld) -> None:
        self.world = world

    def evaluate(self, iterations: int):
        start_positions = self.world.start_positions
        rewards = {pos: [] for pos in start_positions}

        for _ in range(iterations):
            # Sweep every position
            for start_pos in start_positions:
                # Get total reward along path
                reward = self._get_reward(start_pos)
                rewards[start_pos].append(reward)

        # Convert gathered rewards to a grid
        reward_grid = np.zeros(self.world.gridsize)
        for pos, r in rewards.items():
            # The mean of the total rewards
            reward_grid[pos] = np.mean(r)

        self.rewards = reward_grid

    def _get_reward(self, start_pos):
        # Check if in terminal state
        terminal = start_pos in self.world.terminals

        total_reward = 0
        pos = start_pos

        # Take steps until we reach terminal state
        while not terminal:
            pos, reward, terminal = self.world.step(pos, self.choose_action())
            total_reward += reward

        return total_reward

    def choose_action(self):
        # Randomly choose an action
        return random.choice(self.world.actions)

    def plot_rewards(self, ax=None):
        # Check if rewards are set
        assert hasattr(self, "rewards"), "No rewards set yet, run evaluate first."

        # Plot rewards on grid
        ax = sns.heatmap(self.rewards, cmap="RdYlGn", center=0, ax=ax)
        ax.set_title(f"Rewards using Monte Carlo Sweep, mean = {self.rewards.mean():.2f}")

        self.world.plot_walls(ax=ax)

        return ax


class GreedySARSA:
    def __init__(
        self,
        world: GridWorld,
        epsilon: float = 0.1,
        alpha: float = 0.2,
        gamma: float = 0.9,
    ) -> None:
        self.world = world
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def evaluate(self, episodes: int):
        # Keep track of the runtime
        runtime = perf_counter()

        # Initialize Q-table with zeros
        self.q = np.zeros((*self.world.gridsize, len(self.world.actions)))

        # Store simple overview of rewards
        rewards = np.zeros(self.world.gridsize)

        for _ in range(episodes):
            start_pos = random.choice(self.world.start_positions)
            rewards[start_pos] = self._steps(start_pos)

        self.rewards = rewards

        # Store runtime when finished
        self.runtime = perf_counter() - runtime

    def choose_action(self, pos: Tuple[int, int]) -> Tuple[Tuple, int]:
        # Choose action greedily with chance defined by epsilon
        if np.random.uniform(0, 1) > self.epsilon:
            q = self.q[pos]
            (best_idx,) = np.where(q == q.max())

            if len(best_idx) > 1:
                best_idx = np.random.choice(best_idx)
            else:
                best_idx = int(best_idx)

            return self.world.actions[best_idx], best_idx

        # Choose random action
        actions = self.world.actions
        action_idx = np.random.choice(np.arange(len(actions)))

        return actions[action_idx], action_idx

    def _update(
        self,
        pos: Tuple,
        action_idx: int,
        reward: float,
        new_pos: Tuple,
        new_action_idx: int,
    ) -> None:
        curr = *pos, action_idx  # the index of the current in q
        new = *new_pos, new_action_idx  # the index of new in q

        # Update q with SARSA formula
        self.q[curr] += self.alpha * (reward + self.gamma * self.q[new] - self.q[curr])

    def _steps(self, start_pos):
        # Check if current pos is in terminal
        terminal = start_pos in self.world.terminals

        total_reward = 0

        # Determine first pos and action
        pos = start_pos
        action, action_idx = self.choose_action(pos)

        # Keep making steps until terminal state
        while not terminal:
            new_pos, reward, terminal = self.world.step(pos, action)
            new_action, new_action_idx = self.choose_action(new_pos)

            self._update(pos, action_idx, reward, new_pos, new_action_idx)

            pos = new_pos
            action, action_idx = new_action, new_action_idx

            total_reward += reward

        return total_reward

    def plot_rewards(self, ax=None, path=True):
        # Plot the rewards on a grid
        assert hasattr(self, "rewards"), "No rewards set yet, run evaluate first."

        reward_grid = self.rewards

        # Heatmap of rewards
        ax = sns.heatmap(reward_grid, cmap="RdYlGn", center=0, ax=ax)
        ax.set_title(f"Rewards using {str(self)}, mean = {reward_grid.mean():.2f}")

        # Plot arrows in step directions
        if path:
            for pos in set(self.world.start_positions) - set(self.world.terminals):
                q = self.q[pos]
                # Determine best actions
                (best_idx,) = np.where(q == q.max())

                # Draw arrow for every best direction
                for idx in best_idx:
                    action = self.world.actions[idx]
                    ax.arrow(
                        pos[1] + 0.5,
                        pos[0] + 0.5,
                        action[1] * 0.8,
                        action[0] * 0.8,
                        length_includes_head=True,
                        head_width=0.3,
                    )

        self.world.plot_walls(ax=ax)

        return ax

    def __str__(self) -> str:
        return "Greedy SARSA"


class QLearning(GreedySARSA):
    # The only difference to Greedy SARSA is the updating rule, so we just inherit the whole class
    # and overwrite the updating method.

    def _update(self, pos, action_idx, reward, new_pos, *_):
        curr = *pos, action_idx  # the index of the current in q
        new_action_idx = self.q[new_pos].argmax()  # the difference to SARSA
        new = *new_pos, new_action_idx  # the index of the new in q

        # Update q with Q-Learning formula
        self.q[curr] += self.alpha * (reward + self.gamma * self.q[new] - self.q[curr])

    def __str__(self) -> str:
        return "Q-Learning"
