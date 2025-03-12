---
author: Saurav
date: '2023-04-17T16:03:45+0100'
title: Grokking CFR 
# subtitle: A CSS library for creating beautiful Tufte-inspired HTML documents.
meta: true
math: true
toc: false
draft: true
hideDate: true
hideReadTime: true
categories: [templates]
description: "If the description field is not empty, its contents will show in the home page instead of the first 140 characters of the post."
---
CFR is an iterative algorithm designed to compute Nash equilibria in extensive-form games with imperfect information. A Nash equilibrium is a strategy profile where no player can improve their payoff by unilaterally deviating from their strategy. Unlike perfect-information games (e.g., chess), where backward induction or minimax can suffice, imperfect-information games require handling uncertainty over hidden states, modeled via *information sets*. An information set groups game states that a player cannot distinguish given their knowledge.

The "counterfactual" in CFR refers to evaluating actions not taken, asking: "What would my payoff have been had I chosen differently, given what I now know?" The "regret" is the difference between the payoff of the optimal action in hindsight and the action actually taken. CFR minimizes this regret over iterations, converging to an equilibrium strategy.

---

## The Math Behind CFR

Let’s formalize CFR with some notation:

- **Game Tree**: An extensive-form game is represented as a tree with nodes (game states), edges (actions), and leaves (terminal payoffs). Players take turns, and some nodes belong to a chance player (e.g., card deals).
- **Information Set \( I \)**: For player \( i \), \( I \in \mathcal{I}_i \) is a set of game states indistinguishable to \( i \) based on their observations.
- **Actions \( A(I) \)**: The set of legal actions available at information set \( I \).
- **Strategy \( \sigma_i(I, a) \)**: A probability distribution over actions \( a \in A(I) \) for player \( i \) at \( I \), where \( \sum_{a \in A(I)} \sigma_i(I, a) = 1 \).
- **Reach Probability \( \pi_\sigma(h) \)**: The probability of reaching game state \( h \) under strategy profile \( \sigma \), factoring in all players’ strategies and chance events.
- **Counterfactual Reach Probability \( \pi^{-i}_\sigma(h) \)**: The reach probability of \( h \) excluding player \( i \)’s contribution—i.e., the probability of \( h \) occurring if \( i \) had always acted to reach \( h \).

### Utility and Counterfactual Value

For a terminal state \( z \) with payoff \( u_i(z) \) for player \( i \), the expected utility under strategy \( \sigma \) is:

\[ u_i(\sigma) = \sum_{z \in Z} u_i(z) \pi_\sigma(z), \]

where \( Z \) is the set of terminal states.

The counterfactual value of an information set \( I \) for player \( i \) is the expected payoff assuming \( i \) plays to reach \( I \), weighted by the counterfactual reach probability:

\[ v_i(I, \sigma) = \sum_{h \in I} \sum_{z \in Z_h} u_i(z) \pi^{-i}*\sigma(h) \pi*\sigma(h, z), \]

where \( Z_h \) is the set of terminal states reachable from \( h \), and \( \pi_\sigma(h, z) \) is the probability of reaching \( z \) from \( h \) under \( \sigma \).

For a specific action \( a \in A(I) \), the counterfactual value is:

\[ v_i(I, a, \sigma) = \sum_{h \in I} \sum_{z \in Z_{h,a}} u_i(z) \pi^{-i}*\sigma(h) \pi*\sigma(h_a, z), \]

where \( h_a \) is the state reached by taking action \( a \) from \( h \), and \( Z_{h,a} \) are terminal states reachable from \( h_a \).

### Regret Definition

The immediate regret for action \( a \) at \( I \) is the difference between the counterfactual value of taking \( a \) and the value of the current strategy:

\[ r_i(I, a, t) = v_i(I, a, \sigma_t) - v_i(I, \sigma_t), \]

where \( t \) denotes the iteration. The cumulative regret up to iteration \( T \) is:

\[ R_i^T(I, a) = \sum_{t=1}^T r_i(I, a, t). \]

CFR ensures \( R_i^T(I, a) \) grows sub linearly, meaning regret per iteration approaches zero, driving the strategy toward equilibrium.

### Strategy Update with Regret Matching

CFR uses regret matching to update strategies. The cumulative positive regret for action \( a \) is:

\[ R_i^{T,+}(I, a) = \max(R_i^T(I, a), 0). \]

The strategy for the next iteration \( T+1 \) is:

\[ \sigma_i^{T+1}(I, a) = \begin{cases}
\frac{R_i^{T,+}(I, a)}{\sum_{a' \in A(I)} R_i^{T,+}(I, a')} & \text{if } \sum_{a'} R_i^{T,+}(I, a') > 0, \\
\frac{1}{|A(I)|} & \text{otherwise}.
\end{cases} \]

This assigns probabilities proportional to positive regret, defaulting to uniform if no positive regret exists.

### Convergence

The average strategy over \( T \) iterations, \( \bar{\sigma}*i(I, a) = \frac{1}{T} \sum*{t=1}^T \sigma_i^t(I, a) \), converges to a Nash equilibrium as \( T \to \infty \), with the exploitability bounded by \( \frac{\max_I |A(I)| \cdot \Delta_u}{\sqrt{T}} \), where \( \Delta_u \) is the maximum utility range.

---

## A Simple Example: Kuhn Poker

Kuhn Poker is a simplified poker game with a three-card deck (Jack, Queen, King), one card per player, and two actions: *Pass* or *Bet*. It’s small enough to implement CFR from scratch yet rich enough to demonstrate imperfect information.

### Game Rules

- Players ante 1 chip.
- Each gets one card (private).
- Player 1 acts first: Pass or Bet (add 1 chip).
- If Pass, Player 2 acts: Pass (showdown) or Bet (add 1 chip).
- If Bet, opponent can Pass (fold, losing ante) or Call (match bet, showdown).
- Showdown: Higher card wins the pot.

Information sets arise from private cards (e.g., Player 1 with Jack can’t distinguish Player 2’s card).

---

## Python Implementation

Below is a Python implementation of CFR for Kuhn Poker:

```python
import numpy as np
from collections import defaultdict

# Constants
CARDS = ['J', 'Q', 'K']
ACTIONS = ['P', 'B']  # Pass, Bet
NUM_ACTIONS = len(ACTIONS)

class KuhnPoker:
    def __init__(self):
        self.node_map = defaultdict(lambda: {'regret_sum': np.zeros(NUM_ACTIONS),
                                           'strategy_sum': np.zeros(NUM_ACTIONS)})
        self.iterations = 0

    def get_info_set(self, card, history):
        return card + history

    def is_terminal(self, history):
        return history in ['PP', 'PBP', 'PBB', 'BP', 'BB']

    def get_payoff(self, history, cards):
        if history == 'PP' or history == 'BB' or history == 'PBB':
            # Showdown
            pot = 2 if history == 'PP' else 3
            winner = 1 if CARDS.index(cards[0]) > CARDS.index(cards[1]) else -1
            return winner * pot / 2
        elif history == 'BP':
            return 1  # Player 2 folds
        elif history == 'PBP':
            return -1  # Player 1 folds
        return 0

    def cfr(self, cards, history, reach_probs):
        if self.is_terminal(history):
            return self.get_payoff(history, cards)

        player = len(history) % 2
        info_set = self.get_info_set(cards[player], history)
        node = self.node_map[info_set]

        # Get current strategy via regret matching
        strategy = self.get_strategy(node)
        node['strategy_sum'] += reach_probs[player] * strategy

        # Recurse through actions
        util = np.zeros(NUM_ACTIONS)
        node_util = 0
        for i, action in enumerate(ACTIONS):
            next_history = history + action
            new_reach_probs = reach_probs.copy()
            new_reach_probs[player] *= strategy[i]
            util[i] = -self.cfr(cards, next_history, new_reach_probs)
            node_util += strategy[i] * util[i]

        # Update regrets
        for i in range(NUM_ACTIONS):
            regret = util[i] - node_util
            node['regret_sum'][i] += reach_probs[1 - player] * regret

        return node_util

    def get_strategy(self, node):
        regret_sum = node['regret_sum']
        pos_regret_sum = np.maximum(regret_sum, 0)
        total = sum(pos_regret_sum)
        if total > 0:
            return pos_regret_sum / total
        return np.ones(NUM_ACTIONS) / NUM_ACTIONS

    def train(self, iterations):
        util = 0
        for _ in range(iterations):
            self.iterations += 1
            cards = np.random.choice(CARDS, 2, replace=False)
            util += self.cfr(cards, '', [1.0, 1.0])
        return util / iterations

    def get_average_strategy(self, info_set):
        node = self.node_map[info_set]
        strat_sum = node['strategy_sum']
        total = sum(strat_sum)
        if total > 0:
            return strat_sum / total
        return np.ones(NUM_ACTIONS) / NUM_ACTIONS

# Run CFR
game = KuhnPoker()
iterations = 10000
avg_utility = game.train(iterations)
print(f"Average utility: {avg_utility:.4f}")

# Print strategies
for card in CARDS:
    for history in ['', 'P']:
        info_set = card + history
        strat = game.get_average_strategy(info_set)
        print(f"Info set {info_set}: Pass={strat[0]:.3f}, Bet={strat[1]:.3f}")

```

### Code Explanation

- **Node Representation**: Each information set tracks cumulative regret (`regret_sum`) and strategy sums (`strategy_sum`) for averaging.
- **CFR Recursion**: Computes counterfactual values, updates regrets, and aggregates strategy probabilities.
- **Regret Matching**: Normalizes positive regrets to derive the next strategy.
- **Training**: Iterates over random card deals, accumulating utility and strategies.

Running this for 10,000 iterations yields an average strategy approximating the Nash equilibrium. For example, with a Jack, Player 1 might Pass 80% of the time initially and Bet 20%, refining over iterations.