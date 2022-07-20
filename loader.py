from numpy import loadtxt
from tensorflow.keras.models import load_model


def init():
    model = load_model(r"C:\Users\QueenHenny\PycharmProjects\Cats_vs_Dogs\model\model.h5")
    return model
