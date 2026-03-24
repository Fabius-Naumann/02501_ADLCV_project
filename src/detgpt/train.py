from detgpt.data import MyDataset
from detgpt.model import Model


def train():
    dataset = MyDataset("data/raw")
    model = Model()
    # add rest of your training code here


if __name__ == "__main__":
    train()
