import json
import os
import pytest
from sklearn.linear_model import LogisticRegression
from src.train import load_config, train_model
from sklearn.datasets import load_digits

CONFIG_PATH = "config/config.json"

def test_config_file_exists():
    assert os.path.exists(CONFIG_PATH), "Config file not found"

def test_config_structure():
    config = load_config(CONFIG_PATH)
    assert isinstance(config["C"], float), "C should be float"
    assert isinstance(config["solver"], str), "solver should be string"
    assert isinstance(config["max_iter"], int), "max_iter should be int"

def test_train_model_output():
    config = load_config(CONFIG_PATH)
    digits = load_digits()
    X, y = digits.data, digits.target
    model = train_model(X, y, config)
    assert isinstance(model, LogisticRegression), "Model should be LogisticRegression"
    assert hasattr(model, "coef_"), "Model is not fitted properly"

def test_accuracy_threshold():
    config = load_config(CONFIG_PATH)
    digits = load_digits()
    X, y = digits.data, digits.target
    model = train_model(X, y, config)
    accuracy = model.score(X, y)
    assert accuracy > 0.85, f"Accuracy too low: {accuracy}"
