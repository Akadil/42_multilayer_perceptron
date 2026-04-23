import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a multilayer perceptron.")
    parser.add_argument("--dataset", required=True, help="Path to the training CSV file.")
    parser.add_argument(
        "--layer", nargs="+", type=int, default=[24, 24], help="Hidden layer sizes."
    )
    parser.add_argument(
        "--optimization", default="gradient_descent", help="Optimization algorithm to use."
    )
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.0314)
    parser.add_argument("--validation_split", type=float, default=0.2)
    parser.add_argument("--output", default="saved_model.npy", help="Path to save the model.")

    return parser.parse_args()
