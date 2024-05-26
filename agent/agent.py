import json
import os
import random

from .state import State


class Q_State(State):
    '''Augments the game state with Q-learning information'''

    def __init__(self, string):
        super().__init__(string)

        # key stores the state's key string (see notes in _compute_key())
        self.key = self._compute_key()

    def _compute_key(self):
        '''
        Returns a key used to index this state.

        The key should reduce the entire game state to something much smaller
        that can be used for learning. When implementing a Q table as a
        dictionary, this key is used for accessing the Q values for this
        state within the dictionary.
        '''

        # this simple key uses the 3 object characters above the frog
        # and combines them into a key string
        return ''.join([
            self.get(self.frog_x - 1, self.frog_y - 1) or '_',
            self.get(self.frog_x, self.frog_y - 1) or '_',
            self.get(self.frog_x + 1, self.frog_y - 1) or '_',
        ])

    def reward(self):
        '''Returns a reward value for the state.'''

        if self.at_goal:
            return self.score
        elif self.is_done:
            return -10
        else:
            return 0


class Agent:

    def __init__(self, train=None):

        self.prev_state = None
        self.prev_action = None
        
        # train is either a string denoting the name of the saved
        # Q-table file, or None if running without training
        self.train = train

        # q is the dictionary representing the Q-table
        self.q = {}

        # name is the Q-table filename
        # (you likely don't need to use or change this)
        self.name = train or 'q'

        # path is the path to the Q-table file
        # (you likely don't need to use or change this)
        self.path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), 'train', self.name + '.json')

        self.load()

    def load(self):
        '''Loads the Q-table from the JSON file'''
        try:
            with open(self.path, 'r') as f:
                self.q = json.load(f)
            if self.train:
                print('Training {}'.format(self.path))
            else:
                print('Loaded {}'.format(self.path))
        except IOError:
            if self.train:
                print('Training {}'.format(self.path))
            else:
                raise Exception('File does not exist: {}'.format(self.path))
        return self

    def save(self):
        '''Saves the Q-table to the JSON file'''
        with open(self.path, 'w') as f:
            json.dump(self.q, f)
        return self

    def choose_action(self, state_string):
        self.alpha = 0.1  #learning rate and adaptability
        self.gamma = 0.9  #furture sight in planning and decision making
        self.epsilon = 0.2 #rate of which it randomly explores different choices
        current_state = Q_State(state_string)
        
        if current_state.key not in self.q:
            self.q[current_state.key] = {action: 0 for action in State.ACTIONS}

        #Q(Sprev,Aprev) <- (1 - a)Q(Sprev,Aprev)+a[R+ y max a'Q(S, A)
        if self.train and (self.prev_state is not None and self.prev_action is not None):
            reward = current_state.reward()
            old_value = self.q[self.prev_state.key].get(self.prev_action, 0)
            future_rewards = max(self.q[current_state.key].values()) if current_state.key in self.q else 0
            new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * future_rewards)
            self.q[self.prev_state.key][self.prev_action] = new_value
            self.save()

        if self.train and random.random() < self.epsilon:
            action = random.choice(State.ACTIONS)
        else:
            q_values = self.q.get(current_state.key, {})
            action = max(q_values, key=q_values.get) if q_values else random.choice(State.ACTIONS)

        self.prev_state = current_state
        self.prev_action = action

        return action
