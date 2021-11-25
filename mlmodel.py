from scores import scores
from sklearn.base import clone

class mlmodel:
    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.scores = scores()

    # all methods and attributes are passed to the estimator object
    def __getattr__(self, attr: str):

        # just to be pickable
        if attr.startswith('__') and attr.endswith('__'):
            raise AttributeError

        return getattr(self.model, attr)

    def __repr__(self) -> str:
        return f'{self.name}\nscores:{str(self.scores.scores)}'

    def __str__(self) -> str:
        return f'{self.name}\nscores:{str(self.scores.scores)}'

def mlclone(mlmodel_: mlmodel) -> mlmodel:

    assert isinstance(mlmodel_, mlmodel)

    return mlmodel(clone(mlmodel_.model), str(mlmodel_.name))

if __name__ == '__main__':
    print('ML model class file.')