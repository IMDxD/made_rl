from math import *
import random
import numpy as np
from tictactoe import TicTacToe


class Node:
    def __init__(self, move=None, parent=None, empty_spaces=None, turn=None):
        self.move = move
        self.parentNode = parent
        self.childNodes = {}
        self.wins = 0
        self.visits = 0
        self.untriedMoves = list(map(tuple, empty_spaces))
        self.turn = turn

    @staticmethod
    def utc(node, visits):
        return node.wins / node.visits + sqrt(2 * log(visits) / node.visits)

    def utc_select_child(self):
        s = max(self.childNodes.values(), key=lambda x: self.utc(x, self.visits))
        return s

    def random_select(self):
        s = random.choice(list(self.childNodes.values()))
        return s

    def add_child(self, move, empty_spaces, turn):

        node = Node(
            move=move,
            parent=self,
            empty_spaces=empty_spaces,
            turn=turn,
        )
        self.untriedMoves.remove(move)
        self.childNodes[move] = node
        return node

    def update(self, result):
        self.visits += 1
        self.wins += result * self.turn

    def update_untried_moves(self, state, empty_spaces):
        if state not in self.untriedMoves:
            self.untriedMoves[state] = list(map(tuple, empty_spaces))


def state_to_int(state):
    return int(''.join(['%s' % (x + 1) for x in state.ravel()]), 3)


def get_empty_spaces(state, done):
    res = np.where(state == 0)
    if len(res[0]) == 0 or done:
        return []
    else:
        return [(i, j) for i, j in zip(res[0], res[1])]


def rollouts(env, policy_cross, policy_zeros):
    reward = 0
    done = False
    clone_env = TicTacToe(3, 3, 3, clone=env)
    while not done:
        state = state_to_int(env.board)
        if env.curTurn == 1 and state in policy_cross:
            move = policy_cross[state]
        elif env.curTurn == 0 and state in policy_zeros:
            move = policy_zeros[state]
        else:
            move = random.choice(clone_env.getEmptySpaces())
        _, reward, done, _ = clone_env.step(move)
    return reward


def simulate_games(env, policy, turn, iterations=1000):
    env = TicTacToe(3, 3, 3, clone=env)
    rewards = []
    root = policy
    for _ in range(iterations):
        reward = 0
        env.reset()
        done = False
        policy_exists = True
        policy = root
        while not done:
            if env.curTurn == turn and len(policy.childNodes) > 0 and policy_exists:
                policy = max(policy.childNodes.values(), key=lambda x: x.wins / x.visits)
                _, reward, done, _ = env.step(policy.move)
            elif env.curTurn == turn:
                move = random.choice(get_empty_spaces(env.board, env.gameOver))
                _, reward, done, _ = env.step(move)
            else:
                move = random.choice(get_empty_spaces(env.board, env.gameOver))
                _, reward, done, _ = env.step(move)
                if move in policy.childNodes:
                    policy = policy.childNodes[move]
                else:
                    policy_exists = False
        rewards.append(reward)
    return np.mean(rewards)


def uct(env: TicTacToe, itermax, policy_cross, policy_zeros):

    rootnode = Node(empty_spaces=env.getEmptySpaces(), turn=0)
    cross_rewards = []
    zeros_rewards = []
    iterations = []
    for i in range(itermax):
        reward = 0
        done = False
        env.reset()
        node = rootnode

        # select leaf
        while len(node.untriedMoves) == 0 and len(node.childNodes) > 0:
            node = node.utc_select_child()
            _, reward, done, _ = env.step(node.move)

        # expand leaf
        if len(node.untriedMoves) > 0:
            move = random.choice(node.untriedMoves)
            cur_turn = env.curTurn
            _, reward, done, _ = env.step(move)
            node = node.add_child(
                move=move,
                empty_spaces=get_empty_spaces(env.board, env.gameOver),
                turn=cur_turn
            )

        # rollout
        if not done:
            reward = rollouts(env, policy_cross, policy_zeros)

        # backprop
        while node is not None:
            node.update(reward)
            node = node.parentNode

        if (i + 1) % 20000 == 0:
            cross_reward = simulate_games(env, rootnode, 1)
            zeros_reward = -simulate_games(env, rootnode, -1)
            cross_rewards.append(cross_reward)
            zeros_rewards.append(zeros_reward)
            iterations.append(i + 1)
            print(f"Iteration {i + 1}, cross reward: {cross_reward}, zeros reward: {zeros_reward}")

    return rootnode, iterations, cross_rewards, zeros_rewards


def uct_play_game():

    import json
    policy_cross = json.load(open("4_q_policy_cross.json"))
    policy_zeros = json.load(open("4_q_policy_zeros.json"))
    env = TicTacToe(4, 4, 3)
    uct(env=env, itermax=5_000_000, policy_cross=policy_cross, policy_zeros=policy_zeros)


if __name__ == "__main__":
    uct_play_game()
