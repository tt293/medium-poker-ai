from typing import List, Dict
import random
import numpy as np
import sys

Actions = ['B', 'C']  # bet/call vs check/fold
CARDNAMES = ['9', 'T', 'J', 'Q', 'K']

class InformationSet():
    def __init__(self):
        self.cumulative_regrets = np.zeros(shape=len(Actions))
        self.strategy_sum = np.zeros(shape=len(Actions))
        self.num_actions = len(Actions)

    def normalize(self, strategy: np.array) -> np.array:
        """Normalize a strategy. If there are no positive regrets,
        use a uniform random strategy"""
        if sum(strategy) > 0:
            strategy /= sum(strategy)
        else:
            strategy = np.array([1.0 / self.num_actions] * self.num_actions)
        return strategy

    def get_strategy(self, reach_probability: float) -> np.array:
        """Return regret-matching strategy"""
        strategy = np.maximum(0, self.cumulative_regrets)
        strategy = self.normalize(strategy)

        self.strategy_sum += reach_probability * strategy
        return strategy

    def get_average_strategy(self) -> np.array:
        return self.normalize(self.strategy_sum.copy())


class KuhnPoker():
    @staticmethod
    def all_opponents_folded(history: str, num_players: int):
        return len(history) >= num_players and history.endswith('C' * (num_players - 1))

    @staticmethod
    def get_payoff(cards: List[str], history: str, num_players: int) -> List[int]:
        """
        Returns the payoff for all terminal game nodes.
        """
        player = len(history) % num_players
        player_cards = cards[:num_players]
        num_opponents = num_players - 1
        if history == 'C' * num_players:
            payouts = [-1] * num_players
            payouts[np.argmax(player_cards)] = num_opponents
            return payouts
        elif KuhnPoker.all_opponents_folded(history, num_players):
            payouts = [-1] * num_players
            payouts[player] = num_opponents
        else:
            payouts = [-1] * num_players
            active_cards = []
            active_indices = []
            for (ix, x) in enumerate(player_cards):
                if 'B' in history[ix::num_players]:
                    payouts[ix] = -2
                    active_cards.append(x)
                    active_indices.append(ix)
            payouts[active_indices[np.argmax(active_cards)]] = len(active_cards) - 1 + num_opponents
        return payouts

    @staticmethod
    def is_terminal(history: str, num_players: int) -> bool:
        """
        Checks if a given history corresponds to a terminal state
        """
        all_raise = history.endswith('B' * num_players)
        all_acted_after_raise = (history.find('B') > -1) and (len(history) - history.find('B') == num_players)
        all_but_1_player_folds = KuhnPoker.all_opponents_folded(history, num_players)
        return all_raise or all_acted_after_raise or all_but_1_player_folds


class KuhnCFRTrainer():
    def __init__(self, num_players: int):
        self.infoset_map: Dict[str, InformationSet] = {}
        self.num_players = num_players
        self.cards = [x for x in range(self.num_players + 1)]

    def reset(self):
        """reset strategy sums"""
        for n in self.infoset_map.values():
            n.strategy_sum = np.zeros(n.num_actions)

    def get_information_set(self, card_and_history: str) -> InformationSet:
        """add if needed and return"""
        if card_and_history not in self.infoset_map:
            self.infoset_map[card_and_history] = InformationSet()
        return self.infoset_map[card_and_history]

    def get_counterfactual_reach_probability(self, probs: np.array, player: int):
        """compute counterfactual reach probability"""
        return np.prod(probs[:player]) * np.prod(probs[player + 1:])

    def cfr(self, cards: List[str], history: str, reach_probabilities: np.array, active_player: int) -> np.array:
        if KuhnPoker.is_terminal(history, self.num_players):
            return KuhnPoker.get_payoff(cards, history, self.num_players)

        my_card = cards[active_player]
        info_set = self.get_information_set(str(my_card) + history)

        strategy = info_set.get_strategy(reach_probabilities[active_player])
        next_player = (active_player + 1) % self.num_players

        counterfactual_values = [None] * len(Actions)

        for ix, action in enumerate(Actions):
            action_probability = strategy[ix]

            # compute new reach probabilities after this action
            new_reach_probabilities = reach_probabilities.copy()
            new_reach_probabilities[active_player] *= action_probability

            # recursively call cfr method, next player to act is the opponent
            counterfactual_values[ix] = self.cfr(cards, history + action, new_reach_probabilities, next_player)

        # Value of the current game state is just counterfactual values weighted by action probabilities
        node_values = strategy.dot(counterfactual_values)  # counterfactual_values.dot(strategy)
        for ix, action in enumerate(Actions):
            cf_reach_prob = self.get_counterfactual_reach_probability(reach_probabilities, active_player)
            regrets = counterfactual_values[ix][active_player] - node_values[active_player]
            info_set.cumulative_regrets[ix] += cf_reach_prob * regrets
        return node_values

    def train(self, num_iterations: int) -> int:
        utils = np.zeros(self.num_players)
        for _ in range(num_iterations):
            cards = random.sample(self.cards, self.num_players)
            history = ''
            reach_probabilities = np.ones(self.num_players)
            utils += self.cfr(cards, history, reach_probabilities, 0)
        return utils

def pretty_print_infoset_name(name: str, num_players: int) -> str:
    return CARDNAMES[-num_players - 1:][int(name[0])] + name[1:]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        num_iterations = 20000
    else:
        num_iterations = int(sys.argv[1])

    np.set_printoptions(precision=2, floatmode='fixed', suppress=True)
    np.random.seed(43)

    num_players = 3
    cfr_trainer = KuhnCFRTrainer(num_players)

    print(f"\nRunning chance sampling CFR for {num_iterations/10} iterations")
    cfr_trainer.train(int(num_iterations / 10))

    print(f"\nResetting strategy sums")
    cfr_trainer.reset()

    print(f"\nRunning Kuhn Poker chance sampling CFR for {num_iterations} iterations")
    utils = cfr_trainer.train(num_iterations)

    print(f"\nExpected average game value (for player 1): ({(-3/48):.3f}, {(-1./48):.3f})")
    print(f"Expected average game value (for player 2): {(-1./48):.3f}")
    print(f"Expected average game value (for player 3): ({(2./48):.3f}, {(4./48):.3f})\n")

    for player in range(num_players):
        print(f"Computed average game value for player {player + 1}: {(utils[player] / num_iterations):.3f}")

    print(f"History  Bet  Pass")
    for name, info_set in sorted(cfr_trainer.infoset_map.items(), key=lambda s: len(s[0])):
        print(f"{pretty_print_infoset_name(name, num_players):3}:    {info_set.get_average_strategy()}")