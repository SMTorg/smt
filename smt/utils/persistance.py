import pickle


def save(self, filename):
    try:
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
            print("model saved")
    except:
        print("Couldn't save the model")

@staticmethod
def load(filename):
    with open(filename, "rb") as file:
        sm = pickle.load(file)
        return sm