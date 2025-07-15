import pickle
from sklearn.datasets import load_digits

def load_model(path="model_train.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def run_inference(model, X):
    return model.predict(X)

if __name__ == "__main__":
    digits = load_digits()
    X, y = digits.data, digits.target
    model = load_model()
    predictions = run_inference(model, X)
    print("Predictions:", predictions[:10])  # Show first 10 predictions
