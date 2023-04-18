import json
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import QuantileRegressor
from xgboost import XGBRegressor
import argparse
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", "--train_data", required=True)
    parser.add_argument("-out", "--output_file", required=True)
    parser.add_argument("-model", type=str, default="QuantileRegression")
    parser.add_argument("-config", "--model_arg_json", required=True)
    args = parser.parse_args()

    with open(args.model_arg_json) as f:
        config = json.load(f)

    with open(args.train_data, "rb") as f:
        n_feature, n_reward = pickle.load(f)

    n_feature = n_feature.numpy().reshape(len(n_reward), -1)
    n_reward = n_reward.numpy()
    x_train, x_test, y_train, y_test = train_test_split(n_feature, n_reward, test_size=0.2, random_state=42)

    if args.model == "QuantileRegression":
        model = QuantileRegressor(**config["QuantileRegression"])
    elif args.model == "RandomForest":
        model = RandomForestRegressor(**config["RandomForest"])
    elif args.model == "XGBRegression":
        model = XGBRegressor(**config["XGBRegression"])
    else:
        print("Error model name")
        exit(1)

    print("Training start...")
    model.fit(x_train, y_train.ravel())

    print("Prediction start...")
    y_pred = model.predict(x_test)

    print(
        f"""Training error (in-sample performance)
        {model.__class__.__name__}:
        MAE = {mean_absolute_error(y_test, y_pred):.3f}
        MSE = {mean_squared_error(y_test, y_pred):.3f}
        """
    )

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "wb") as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    main()
