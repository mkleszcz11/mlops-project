from tsai.basics import *
from models.model import Forecaster

def train():
    
    # TODO add arguments to allow parameter tuning
    model = Forecaster()
    model.train_model()

if __name__ == "__main__":
    train()
