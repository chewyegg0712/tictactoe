from collections import namedtuple
from random import choice
import matplotlib.pyplot as plt
import networkx as nx
from monte_carlo_tree_search import MCTS, Node

_TTTB = namedtuple("TicTacToeBoard", "tup turn winner terminal")


class TicTacToeBoard(_TTTB, Node):
    def find_children(self):
        if self.terminal:
            return set()
        return {
            self.make_move(i) for i, value in enumerate(self.tup) if value is None
        }

    def find_random_child(self):
        if self.terminal:
            return None
        empty_spots = [i for i, value in enumerate(self.tup) if value is None]
        return self.make_move(choice(empty_spots))

    def reward(self):
        if not self.terminal:
            raise RuntimeError(f"reward called on nonterminal board {self}")
        if self.winner is self.turn:
            raise RuntimeError(f"reward called on unreachable board {self}")
        if self.turn is (not self.winner):
            return 0
        if self.winner is None:
            return 0.5
        raise RuntimeError(f"board has unknown winner type {self.winner}")

    def is_terminal(self):
        return self.terminal

    def make_move(self, index):
        tup = self.tup[:index] + (self.turn,) + self.tup[index + 1:]
        turn = not self.turn
        winner = _find_winner(tup)
        is_terminal = (winner is not None) or not any(v is None for v in tup)
        return TicTacToeBoard(tup, turn, winner, is_terminal)

    def to_pretty_string(self):
        to_char = lambda v: ("X" if v is True else ("O" if v is False else " "))
        rows = [
            [to_char(self.tup[3 * row + col]) for col in range(3)] for row in range(3)
        ]
        return (
                "\n  1 2 3\n"
                + "\n".join(str(i + 1) + " " + " ".join(row) for i, row in enumerate(rows))
                + "\n"
        )


def _winning_combos():
    for start in range(0, 9, 3):
        yield (start, start + 1, start + 2)
    for start in range(3):
        yield (start, start + 3, start + 6)
    yield (0, 4, 8)
    yield (2, 4, 6)


def _find_winner(tup):
    for i1, i2, i3 in _winning_combos():
        v1, v2, v3 = tup[i1], tup[i2], tup[i3]
        if False is v1 is v2 is v3:
            return False
        if True is v1 is v2 is v3:
            return True
    return None


def new_tic_tac_toe_board():
    return TicTacToeBoard(tup=(None,) * 9, turn=True, winner=None, terminal=False)


def play_game():
    tree = MCTS()
    board = new_tic_tac_toe_board()
    print(board.to_pretty_string())

    while True:
        # Player's turn
        while True:
            try:
                row_col = input("Your turn! Enter row,col (e.g., 1,1): ")
                row, col = map(int, row_col.split(","))
                index = 3 * (row - 1) + (col - 1)
                if board.tup[index] is not None:
                    print("Invalid move! That spot is already taken.")
                    continue
                break
            except ValueError:
                print("Invalid input! Please enter row,col (e.g., 1,1).")

        board = board.make_move(index)
        print(board.to_pretty_string())
        visualize_mcts_tree(tree, board)  # Visualize MCTS tree after player's move
        if board.terminal:
            break

        # Computer's turn
        print("Computer's turn...")
        for _ in range(50):
            tree.do_rollout(board)
        board = tree.choose(board)
        print(board.to_pretty_string())
        visualize_mcts_tree(tree, board)  # Visualize MCTS tree after computer's move

        if board.terminal:
            break

    # Game ended, display result
    if board.winner is True:
        print("Congratulations! You won!")
    elif board.winner is False:
        print("Sorry, you lost. Better luck next time!")
    else:
        print("It's a draw!")

    # Ask if the player wants to play again
    play_again = input("Do you want to play again? (yes/no): ").lower()
    if play_again == "yes":
        play_game()
    else:
        print("Thanks for playing!")


def visualize_mcts_tree(tree, root_node):
    G = nx.DiGraph()
    node_labels = {}

    def add_nodes(node):
        if node not in node_labels:
            node_labels[node] = f"Q: {tree.Q[node]}, N: {tree.N[node]}"
            G.add_node(node_labels[node])
            for child in tree.children.get(node, []):
                child_label = node_labels.get(child, None)
                if child_label is None:
                    child_label = f"Q: {tree.Q[child]}, N: {tree.N[child]}"
                    node_labels[child] = child_label
                    G.add_node(child_label)
                G.add_edge(node_labels[node], child_label)
                add_nodes(child)

    add_nodes(root_node)

    pos = nx.spring_layout(G)

    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, arrows=True)
    plt.show()


if __name__ == "__main__":
    play_game()