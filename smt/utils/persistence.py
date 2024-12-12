import pickle
import zlib


def save(self, filename):
    serialized_data = pickle.dumps(self, protocol=5)
    compressed_data = zlib.compress(serialized_data)
    with open(filename, "wb") as file:
        file.write(compressed_data)


def load(filename):
    with open(filename, "rb") as file:
        compressed_data = file.read()

    serialized_data = zlib.decompress(compressed_data)
    sm = pickle.loads(serialized_data)

    return sm
