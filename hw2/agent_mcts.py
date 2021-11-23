from math import *
import random
from collections import defaultdict
from tictactoe import TicTacToe


class Node:
    def __init__(self, state, move=None, parent=None, empty_spaces=None, turn=None):
        self.move = move  # the move that got us to this node - "None" for the root node
        self.parentNode = parent  # "None" for the root node
        self.childNodes = defaultdict(list)
        self.wins = 0
        self.visits = 0
        if state is None:
            self.untriedMoves = {}
        else:
            self.untriedMoves = {
                state: list(map(tuple, empty_spaces))
            }
        self.turn = turn

    def utc_select_child(self, state):

        s = max(
            self.childNodes[state],
            key=lambda x: x.wins / x.visits + sqrt(2 * log(self.visits) / x.visits),
        )
        return s

    def add_child(self, new_state, state, move, empty_spaces, turn):

        n = Node(
            state=new_state,
            move=move,
            parent=self,
            empty_spaces=empty_spaces,
            turn=turn,
        )
        self.untriedMoves[state].remove(move)
        self.childNodes[state].append(n)
        return n

    def update(self, result):
        self.visits += 1
        self.wins += result

    def update_untried_moves(self, state, empty_spaces):
        if state not in self.untriedMoves:
            self.untriedMoves[state] = list(map(tuple, empty_spaces))


def uct(env: TicTacToe, itermax, turn):

    if turn == 1:
        rootnode = Node(state=env.getHash(), empty_spaces=env.getEmptySpaces(), turn=turn)
    else:
        rootnode = Node(state=None, empty_spaces=env.getEmptySpaces(), turn=turn)

    env = TicTacToe(3, 3, 3)
    for i in range(itermax):
        reward = 0
        env.reset()
        state = env.getHash()
        node = rootnode
        if env.curTurn != rootnode.turn:
            _, reward, _, _ = env.step(random.choice(env.getEmptySpaces()))
            state = env.getHash()
            node.update_untried_moves(state, env.getEmptySpaces())

        # select leaf
        while len(node.untriedMoves[state]) == 0 and len(node.childNodes[state]) > 0:
            node = node.utc_select_child(state)
            _, reward, _, _ = env.step(node.move)
            if not env.gameOver:
                _, reward, _, _ = env.step(random.choice(env.getEmptySpaces()))
                state = env.getHash()
                if env.gameOver:
                    node.update_untried_moves(state, [])
                    break
                else:
                    node.update_untried_moves(state, env.getEmptySpaces())
            else:
                state = env.getHash()
                break

        # expand leaf
        if node.untriedMoves[state]:
            if env.curTurn != rootnode.turn:
                _, reward, _, _ = env.step(random.choice(env.getEmptySpaces()))
                node.update_untried_moves(state, env.getEmptySpaces())
            move = tuple(random.choice(node.untriedMoves[state]))
            _, reward, _, _ = env.step(move)
            if not env.gameOver:
                _, reward, _, _ = env.step(random.choice(env.getEmptySpaces()))
                new_state = env.getHash()
                node = node.add_child(
                    new_state=new_state,
                    state=state,
                    move=move,
                    empty_spaces=env.getEmptySpaces(),
                    turn=env.curTurn,
                )
            else:
                node = node.add_child(
                    new_state=env.getHash(),
                    state=state,
                    move=move,
                    empty_spaces=[],
                    turn=env.curTurn,
                )

        # rollout
        while not env.gameOver:
            _, reward, _, _ = env.step(random.choice(env.getEmptySpaces()))

        # backprop
        while node is not None:
            node.update(reward * rootnode.turn)
            node = node.parentNode

    return rootnode


def uct_play_game():

    env = TicTacToe(3, 3, 3)
    node_cross = uct(env=env, itermax=500_000, turn=1)
    node_zeros = uct(env=env, itermax=500_000, turn=-1)
    env.reset()
    done = False
    while not done:
        print(env.board)
        if env.curTurn == 1:
            node_cross = max(node_cross.childNodes[env.getHash()], key=lambda x: x.wins)
            _, _, done, _ = env.step(node_cross.move)
        else:
            node_zeros = max(node_zeros.childNodes[env.getHash()], key=lambda x: x.wins)
            _, _, done, _ = env.step(node_zeros.move)
    print(env.board)


if __name__ == "__main__":
    uct_play_game()
