
class AbstractClassifier:
    def fit(self, x, y):
        """abstract function
           check what is the x, y
        """
        raise NotImplementedError

    def predict(self, x):
        """abstract function
           check what is the x
        """
        raise NotImplementedError

