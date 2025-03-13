---
author: [Your Name]
date: '2025-03-12T10:00:00+0000'
title: Grokking Counterfactual Regret Minimization
subtitle: A Deep Dive into Theory and Implementation
meta: true
math: true
toc: false
draft: false
hideDate: false
hideReadTime: true
categories: [AI, Game Theory]
description: "Unpacking the mathematics of Counterfactual Regret Minimization and implementing it from scratch in Python for a graduate audience."
---
{{< epigraph author="John von Neumann" cite="Theory of Games and Economic Behavior" >}}
Real life consists of bluffing, of little tactics of deception, of asking yourself what is the other man going to think I mean to do.
{{< /epigraph >}}

Counterfactual Regret Minimization (CFR) is a cornerstone algorithm in artificial intelligence for solving imperfect-information games—scenarios like poker where players operate with incomplete knowledge. Pioneered by Martin Zinkevich et al. in 2007, CFR has propelled AI to superhuman levels in games such as Heads-Up No-Limit Texas Hold'em. This blog elucidates the mathematical underpinnings of CFR and provides a Python implementation from scratch, tailored for graduate students in AI and game theory.

---

## What is CFR?

CFR iteratively computes Nash equilibria in extensive-form games with imperfect information. A Nash equilibrium ensures no player can unilaterally improve their payoff, a concept critical in strategic decision-making. Unlike perfect-information games (e.g., chess), where minimax suffices, imperfect-information games demand handling uncertainty via *information sets*—groupings of indistinguishable game states.

The "counterfactual" aspect evaluates unchosen actions, pondering: "What payoff would I have gained otherwise?" Regret, the difference between this hypothetical payoff and the actual outcome, is minimized over iterations, converging to an equilibrium strategy. {{< sidenote >}}This iterative regret minimization distinguishes CFR from static optimization methods.{{< /sidenote >}}

---

## The Math Behind CFR

{{< newthought >}}Let’s formalize CFR{{< /newthought >}} with key notation:

- **Game Tree**: Represented as nodes (states), edges (actions), and leaves (payoffs), with chance nodes for events like card deals.
- **Information Set \( I \)**: For player \( i \), \( I \in \mathcal{I}_i \) groups states \( i \) cannot distinguish.
- **Actions \( A(I) \)**: Legal moves at \( I \).
- **Strategy \( \sigma_i(I, a) \)**: Probability of action \( a \in A(I) \), where \( \sum_{a \in A(I)} \sigma_i(I, a) = 1 \).
- **Reach Probability \( \pi_{\sigma}(h) \)**: Probability of reaching state \( h \) under strategy \( \sigma \).
- **Counterfactual Reach \( \pi^{-i}_{\sigma}(h) \)**: Reach probability excluding \( i \)’s actions.

### Utility and Counterfactual Value

Expected utility for player \( i \) under strategy \( \sigma \) is:

$$ u_i(\sigma) = \sum_{z \in Z} u_i(z) \pi_{\sigma}(z), $$

where \( Z \) denotes terminal states. The counterfactual value at \( I \) assumes \( i \) plays to reach \( I \):

$$ v_i(I, \sigma) = \sum_{h \in I} \sum_{z \in Z_h} u_i(z) \pi^{-i}_{\sigma}(h) \pi_{\sigma}(h, z). $$

For action \( a \):

`$$ v_i(I, a, \sigma) = \sum_{h \in I} \sum_{z \in Z_{h,a}} u_i(z) \pi^{-i}_{\sigma}(h) \pi_{\sigma}(h_a, z). $$`

### Regret and Updates

Immediate regret for action \( a \) at iteration \( t \) is:

$$ r_i(I, a, t) = v_i(I, a, \sigma_t) - v_i(I, \sigma_t), $$

with cumulative regret:

$$ R_i^T(I, a) = \sum_{t=1}^T r_i(I, a, t). $$

CFR employs regret matching, defining positive regret:

`$$ R_i^{T,+}(I, a) = \max(R_i^T(I, a), 0), $$`

and updating the strategy. 

<p>
\[ \sigma_i^{T+1}(I, a) = \begin{cases} 
\frac{R_i^{T,+}(I, a)}{\sum_{a' \in A(I)} R_i^{T,+}(I, a')} & \text{if } \sum_{a' \in A(I)} R_i^{T,+}(I, a') > 0, \\
\frac{1}{|A(I)|} & \text{otherwise}.
\end{cases} \]
</p>

### Convergence

The average strategy \( \bar{\sigma}_i(I, a) = \frac{1}{T} \sum_{t=1}^T \sigma_i^t(I, a) \) converges to a Nash equilibrium, with exploitability bounded by:

$$ \frac{\max_{I} |A(I)| \cdot \Delta_u}{\sqrt{T}}, $$

where (\Delta_u) is the utility range. {{< marginnote ind="⚠" >}}Convergence is slower in larger games, prompting optimizations like CFR+.{{< /marginnote >}}

---

## Kuhn Poker: A Case Study

Kuhn Poker, a simplified poker variant, uses a three-card deck (Jack, Queen, King) and two actions: *Pass* or *Bet*. It’s ideal for illustrating CFR’s mechanics.

### Rules

- Players ante 1 chip.
- Each receives one private card.
- Player 1 acts: Pass or Bet (add 1 chip).
- Responses vary: Pass leads to showdown or Bet; Bet leads to fold or call.
- Showdown awards the pot to the higher card.

Information sets stem from hidden cards, creating uncertainty ripe for CFR.

---

## Python Implementation

Here’s a from-scratch implementation in Python, leveraging NumPy for efficiency:

```
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
        if history in ['PP', 'BB', 'PBB']:
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

        # Regret matching
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
        return pos_regret_sum / total if total > 0 else np.ones(NUM_ACTIONS) / NUM_ACTIONS

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
        return strat_sum / total if total > 0 else np.ones(NUM_ACTIONS) / NUM_ACTIONS

# Execution
game = KuhnPoker()
iterations = 10000
avg_utility = game.train(iterations)
print(f"Average utility: {avg_utility:.4f}")

for card in CARDS:
    for history in ['', 'P']:
        info_set = card + history
        strat = game.get_average_strategy(info_set)
        print(f"Info set {info_set}: Pass={strat[0]:.3f}, Bet={strat[1]:.3f}")
```

### Code Breakdown

- **Initialization**: Uses a `defaultdict` to store regret and strategy sums for each information set. {{< sidenote >}}Highlighted lines (38-44) implement regret matching, the core of CFR.{{< /sidenote >}}
- **CFR Function**: Recursively computes utilities, updates regrets, and aggregates strategies.
- **Output**: After 10,000 iterations, it prints the average strategy, approximating the Nash equilibrium.

---