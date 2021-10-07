from itertools import count
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import seaborn


def cmp(a, b):
    return float(a > b) - float(a < b)


# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def draw_card(np_random):
    return int(np_random.choice(deck))


def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


class BlackjackEnv(gym.Env):

    def __init__(self, natural=False):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2))
        )
        self.seed()

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        # Start the first game
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        if action:  # hit: add a card to players hand and return
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1.0
            else:
                done = False
                reward = 0.0
        else:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1.0:
                reward = 1.5
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

    def reset(self):
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)
        return self._get_obs()


class BlackjackEnvDouble(gym.Env):

    def __init__(self, natural=False):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2))
        )
        self.seed()

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        # Start the first game
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        if action == 1:  # hit: add a card to players hand and return
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1.0
            else:
                done = False
                reward = 0.0
        elif action == 0:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1.0:
                reward = 1.5
        else:
            done = True
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1.0
            else:
                while sum_hand(self.dealer) < 17:
                    self.dealer.append(draw_card(self.np_random))
                reward = cmp(score(self.player), score(self.dealer))
                if self.natural and is_natural(self.player) and reward == 1.0:
                    reward = 1.5
                reward *= 2
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

    def reset(self):
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)
        return self._get_obs()


class Deck:

    MIN_VALUE = -32

    def __init__(self):
        self.cards = [4, 4, 4, 4, 4, 4, 4, 4, 4, 16]
        self.card_values_map = {
            1: 0,
            2: 1,
            3: 1,
            4: 2,
            5: 2,
            6: 1,
            7: 1,
            8: 0,
            9: 0,
            10: -2
        }
        self._deck_count = 52
        self._deck_value = 0

    def reset_deck(self):
        self.cards = [4, 4, 4, 4, 4, 4, 4, 4, 4, 16]
        self._deck_count = 52
        self._deck_value = 0

    def draw_card(self):
        card_index = np.random.choice([i for i, count in enumerate(self.cards) if count > 0])
        card = card_index + 1
        self.cards[card_index] -= 1
        self._deck_count -= 1
        self._deck_value -= self.card_values_map[card]
        return card

    @property
    def deck_value(self):
        return self._deck_value - self.MIN_VALUE

    @property
    def deck_count(self):
        return self._deck_count



class BlackjackEnvDoubleDeck(gym.Env):

    def __init__(self, natural=False):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple(
            (
                spaces.Discrete(32), 
                spaces.Discrete(11),
                spaces.Discrete(2), 
                spaces.Discrete(65),
                spaces.Discrete(2)
            )
        )
        self.seed()
        self.deck = Deck()

        self.player = []
        self.dealer = []
        self.natural = natural

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        unknown_dealer = True
        if action == 1:  # hit: add a card to players hand and return
            self.player.append(self.deck.draw_card())
            if is_bust(self.player):
                done = True
                reward = -1.0
            else:
                done = False
                reward = 0.0
        elif action == 0:  # stick: play out the dealers hand, and score
            unknown_dealer = False
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(self.deck.draw_card())
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1.0:
                reward = 1.5
        else:
            done = True
            self.player.append(self.deck.draw_card())
            if is_bust(self.player):
                done = True
                reward = -1.0
            else:
                while sum_hand(self.dealer) < 17:
                    self.dealer.append(self.deck.draw_card())
                reward = cmp(score(self.player), score(self.dealer))
                if self.natural and is_natural(self.player) and reward == 1.0:
                    reward = 1.5
                reward *= 2
        return self._get_obs(unknown_dealer), reward, done, {}

    def _get_obs(self, unknown_dealer=True):
        known_value = self.deck.deck_value
        if unknown_dealer:
            known_value += self.deck.card_values_map[self.dealer[1]]
        return (
            sum_hand(self.player), 
            self.dealer[0], 
            usable_ace(self.player),
            known_value,
            unknown_dealer
        )

    def reset(self):
        self.dealer = []
        self.player = []
        if self.deck.deck_count < 15:
            self.deck.reset_deck()
        for _ in range(2):
            self.dealer.append(self.deck.draw_card())
            self.player.append(self.deck.draw_card())
        return self._get_obs()
