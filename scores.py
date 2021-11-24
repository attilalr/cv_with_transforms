from typing import Dict
import numpy as np

class scores(dict):
    def __init__(self):
        self.scores = {}

    def clear(self) -> None:
        '''
        Reset the dictionary which hold the scores.
        '''
        for key in self.scores:
            self.__dict__[key] = list()

        self.scores = {}

    def register(self, name, value) -> None:
        '''
        Register a value to the score key name.
        '''
        if name in self.scores:
            self.scores[name].append(value)
        else:
            print (f'Starting metric {name}.')
            self.scores[name] = list()
            self.scores[name].append(value)

        # to use scores[metric] in place of scores.scores[metric]
        # it is slow...
        for key in self.scores:
            self.__dict__[key] = self.scores[key]

    def __getitem__(self, key):
        return self.__dict__[key]

    def mean(self) -> Dict:
        '''
        Returns a dictionary with a mean value for each score name.
        '''
        d = {}
        for name in self.scores:
            d[name] = np.mean(self.scores[name])
        return d

    def std(self) -> Dict:
        '''
        Returns a dictionary with a std value for each score name.
        '''
        d = {}
        for name in self.scores:
            d[name] = np.std(self.scores[name])
        return d

    def size(self) -> Dict:
        '''
        Returns a dictionary with size value for each score name.
        '''
        d = {}
        for name in self.scores:
            d[name] = np.size(self.scores[name])
        return d

    def __repr__(self) -> str:
        return str(self.scores)

    def __str__(self) -> str:
        return str(self.scores)
