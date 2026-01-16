from src.mlops.model import Model
from src.mlops.data import MyDataset

def train():
    dataset = MyDataset("data/processed")
    model = Model(num_classes= 43)
    # add rest of your training code here

if __name__ == "__main__":
    train()
