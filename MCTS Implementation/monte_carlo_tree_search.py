from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
import math
from typing import Set, Optional
from collections import namedtuple
from random import choice
import matplotlib.pyplot as plt
import networkx as nx

_TTTB = namedtuple("TicTacToeBoard", "tup turn winner terminal")

class MCTS:
    """Monte Carlo tree searcher. First rollout the tree then choose a move."""

    def __init__(self, exploration_weight: float = 1):
        self.Q: dict = defaultdict(int)  # total reward of each node
        self.N: dict = defaultdict(int)  # total visit count for each node
        self.children: dict = {}  # children of each node
        self.exploration_weight: float = exploration_weight

    def choose(self, node: Node) -> Node:
        """Choose the best successor of node. (Choose a move in the game)"""
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n: Node) -> float:
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[node], key=score)

    def do_rollout(self, node: Node) -> None:
        """Make the tree one layer better. (Train for one iteration.)"""
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node: Node) -> list[Node]:
        """Find an unexplored descendent of `node`"""
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node: Node) -> None:
        """Update the `children` dict with the children of `node`"""
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def _simulate(self, node: Node) -> float:
        """Returns the reward for a random simulation (to completion) of `node`"""
        invert_reward = True
        while True:
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            next_node = node.find_random_child()
            if next_node is None:
                raise ValueError("find_random_child() returned None")
            node = next_node
            invert_reward = not invert_reward

    def _backpropagate(self, path: list[Node], reward: float) -> None:
        """Send the reward back up to the ancestors of the leaf"""
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node: Node) -> Node:
        """Select a child of node, balancing exploration & exploitation"""
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n: Node) -> float:
            """Upper confidence bound for trees"""
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)


class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self) -> Set[Node]:
        """All possible successors of this board state"""
        return set()

    @abstractmethod
    def find_random_child(self) -> Optional[Node]:
        """Random successor of this board state (for more efficient simulation)"""
        return None

    @abstractmethod
    def is_terminal(self) -> bool:
        """Returns True if the node has no children"""
        return True

    @abstractmethod
    def reward(self) -> float:
        """Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"""
        return 0

    @abstractmethod
    def __hash__(self) -> int:
        """Nodes must be hashable"""
        return 123456789

    @abstractmethod
    def __eq__(self, other: Node) -> bool:
        """Nodes must be comparable"""
        return True

