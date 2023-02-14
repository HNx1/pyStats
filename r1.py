# A first look at reinforcement learning to solve a basic game
# The game - two players, each player says an integer between 1 and k on each turn, alternating turns.
# Tracking the cumulative sum c of the chosen numbers, the goal is to be the player who makes the play that makes c equal to n
# n and k are known positive integers for the game.
# It's known analytically that this game is won by the first player choosing n (mod (k+1)) (if non-zero) as their first play, then setting c=n(mod(k+1)) at each stage with their play
# This is always possible as if m is the previous play of player 2, player one can always play k+1-m, making c mod(k+1) invariant after each of player 1's plays.
# If n mod(k+1) ==0, then player 2 wins as they effectively become player 1 after player 1's first play  .
# So analytically, player 1 should win 1-1/(k+1) proportion of the time, given random n for fixed k.

# Our agents understand how to play the game for fixed n and k
# It's easy to slightly adapt the code to instead accommodate any n < some fixed A (by perturbing starting state away from zero)
# You can make agents that can play for unbounded n by instead only containing strategy mod (k+1),
# but this so constrains the strategy towards the analytical solution that it is mostly uninteresting as a toy model.
# Another natural unbounded implementation is to reverse the strategy and expectations lists, keeping low remaining state at the front,
# and then extending the needed attributes when a new larger remaining state is encountered.

# AnalyticalAgent always plays the analytical solution.
# Our goal is to train learning agents, who start with random behaviour but always go first.
# We have a basic reward system, where 1 is returned as reward to all actions taken if win, and -1 is returned if lose

# It's unsurprisingly very inefficient as we try to increase the game size, but it can quickly solve the game for relatively low values.
# Learning Agent 2 is more powerful - it will store an expected reward instead of a pure decision probability, and choose on the basis of the expected reward
# It has a reward process very similar to AlphaGoZero, where it directly passes the final game outcome as a reward to each state-action pair that resulted in that
# However, it still basically peters out at relatively low levels

# Learning Agent 3 can also access its opponents moves in action,state,reward. This converges much faster as
# a) it will get twice as many moves per game and, more importantly, b) half the moves will be analytically perfect if playing the analytic agent


# Numpy used only for random choice and some functions (like exp). Other than that is pure python
import numpy as np


class Agent:
    def __init__(self, name, max_play, target):
        self.name = name
        self.max_play = max_play
        self.target = target
        self.learner = False
        self.access = False
        self.strategy = [
            [1/max_play for _ in range(max_play)]for _ in range(target)]

    def __repr__(self):
        return f"Agent {self.name}"

    def play(self, game_state):
        # Pick a value from the strategy, need to add 1 due to picking from 0,1...k-1
        row = self.strategy[game_state]
        choice = np.random.choice(self.max_play, p=row)
        return choice+1


class LearningAgent(Agent):
    def __init__(self, name, max_play, target):
        super().__init__(name, max_play, target)
        self.learner = True
        self.visit_counts = [
            [0 for _ in range(max_play)]for _ in range(target)]

    def learn(self, actions, states, rewards):
        for action, state, reward in zip(actions, states, rewards):
            # Based on reward for action in state, change future probability of that action and renormalise row
            self.strategy[state][action] *= (1+reward*0.1)
            s = sum(self.strategy[state])
            # Renormalise
            self.strategy[state] = [x/s for x in self.strategy[state]]


class LearningAgent2(Agent):
    def __init__(self, name, max_play, target):
        super().__init__(name, max_play, target)
        # Expectation based system, where our choice will be the maximum expectation each time
        self.learner = True
        self.visit_counts = [
            [1 for _ in range(max_play)]for _ in range(target)]
        self.expectations = [
            [0 for _ in range(max_play)]for _ in range(target)]
        self.strategy = [[np.exp(-1/(1+x))/sum(np.exp(-1/(1+xi))
                                               for xi in row) for x in row]for row in self.expectations]

    def learn(self, actions, states, rewards):
        for action, state, reward in zip(actions, states, rewards):
            # Based on reward for action in state, change future probability of that action and renormalise row
            c = self.visit_counts[state][action]
            self.expectations[state][action] *= c
            self.expectations[state][action] += reward
            self.expectations[state][action] /= (c+1)
            self.visit_counts[state][action] += 1
            self.strategy[state] = [
                np.exp(-1/(1+x)) if x > -0.99 else 0 for x in self.expectations[state]]
            s = sum(self.strategy[state])
            # Handle potential zero state, resetting strategy
            if s == 0:
                n = len(self.strategy[state])
                self.strategy[state] = [1/n for _ in range(n)]
            else:
                self.strategy[state] = [x/s for x in self.strategy[state]]


class LearningAgent3(LearningAgent2):
    def __init__(self, name, max_play, target):
        super().__init__(name, max_play, target)
        self.access = True


class AnalyticalAgent(Agent):
    def __init__(self, name, max_play, target):
        super().__init__(name, max_play, target)
        # If there is a winning solution, play it, otherwise play randomly
        for i in range(target):
            if (target-i) % (max_play+1) != 0:
                self.strategy[i] = [
                    1 if (target-j-i-1) % (max_play+1) == 0 else 0 for j in range(max_play)]


# class SemiRandomAgent(Agent):
#     def __init__(self, name, max_play, target):
#         super().__init__(name, max_play, target)
#         self.analytical_strategy = self.strategy
#         # If there is a winning solution, play it, otherwise play randomly
#         for i in range(target):
#             if (target-i) % (max_play+1) != 0:
#                 self.analytical_strategy[i] = [
#                     1 if (target-j-i-1) % (max_play+1) == 0 else 0 for j in range(max_play)]


class TargetGame():
    def __init__(self, agents):
        if len(agents) != 2:
            raise Exception("This is a two player game")
        assert agents[0].target == agents[1].target
        assert agents[0].max_play == agents[1].max_play
        self.agents = agents
        self.state = 0
        self.target = self.agents[0].target
        self.winners = []
        self.win_props = []
        self.access = agents[0].access

    def reset(self):
        self.state = 0
        self.winners = []

    def full_reset(self):
        self.reset()
        self.win_props = []

    def play_game(self):
        # Memo to send to learning agent
        actions, states, actors = [], [], []
        winner = 0
        while self.state < self.target:
            # While we are below target, agents play in order. If the state is hit, that agent wins. If it's above, that agent loses.
            for i, agent in enumerate(self.agents):
                choice = agent.play(self.state)
                if agent.learner or self.access:
                    actions.append(choice-1)
                    states.append(self.state)
                    actors.append(i)
                self.state += choice
                if self.state == self.target:
                    winner = i
                    break
                elif self.state > self.target:
                    winner = 1-i
                    break
        self.winners.append(winner)
        agent = self.agents[0]
        if agent.learner:
            # Rewards in {-1,1}, reward from other agent is reversed from point of view of learning agent
            rewards = [1 if (winner, actors[i]) in [(1, 1), (0, 0)]
                       else -1 for i in range(len(actions))]
            agent.learn(actions, states, rewards)

    def play_batch(self, iters=100, chosen_agent=0):
        # For chosen agent, appends their winning percentage in a set of games to the win_proportions
        self.reset()
        for _ in range(iters):
            self.play_game()
            self.state = 0
        win_prop = sum(
            1 if x == chosen_agent else 0 for x in self.winners)/iters
        self.win_props.append(win_prop)

    def full_run(self, rounds=200, iters=100, chosen_agent=0):
        # Runs a certain number of sets of games, outputting the win percentage of the agent in each set.
        self.full_reset()
        for _ in range(rounds):
            self.play_batch(iters=iters, chosen_agent=chosen_agent)
