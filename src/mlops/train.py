# from .model import Model
from .data.dataset import GTSRB


def train():
    print("testting_train")
    dataset = GTSRB(raw_dir="data/raw/gtsrb", processed_dir="data/processed", mode="train")
    # model = Model(num_classes= 43)
    # add rest of your training code here

if __name__ == "__main__":
    train()

