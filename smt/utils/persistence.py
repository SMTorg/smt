import pickle


def save(self, filename):
    with open(filename, "wb") as file:
        pickle.dump(self, file)


def load(filename):
    sm = None
    with open(filename, "rb") as file:
        sm = pickle.load(file)
    return sm
