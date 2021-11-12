from scores import scores

class mlmodel:
    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.scores = scores()

    # all methods and attributes are passed to the estimator object
    def __getattr__(self, name):
        return getattr(self.model, name)

    def __repr__(self) -> str:
        return f'{self.name}\nscores:{str(self.scores.scores)}'

    def __str__(self) -> str:
        return f'{self.name}\nscores:{str(self.scores.scores)}'

if __name__ == '__main__':
    print('ML model class file.')