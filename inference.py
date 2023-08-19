from flask import Flask, request
from simpletransformers.classification import ClassificationModel
import logging
import pandas as pd
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO)
MODEL_CHECPOINT_PATH = f"{os.path.join(Path(__file__).parent,'content', 'outputs', 'checkpoint')}"

def validate_input(data):
    """
    Generic validation function to check all input datatypes
    :param data:
    """
    # List of expected keys and their types
    expected_keys_types = {
        "map_id": int,
        "map_title": str,
        "map_rating_": int,
        "idea_id": int,
        "idea_parent_id": (type(None), int),  # Can be None or int
        "idea_title": str
    }

    # Check if all required keys are in data
    for row in data:
        for key, expected_type in expected_keys_types.items():
            if key not in row:
                raise ValueError(f"Key '{key}' is missing from the row.")

            if not isinstance(row[key], expected_type):
                if isinstance(expected_type, tuple):  # for multiple acceptable types
                    if not any(isinstance(row[key], t) for t in expected_type):
                        raise ValueError(f"Value for '{key}' is not one of the expected types: {expected_type}.")
                else:
                    raise ValueError(f"Value for '{key}' is not of the expected type: {expected_type}.")

    # Additional checks (e.g. length checks or value range checks) can be added if necessary

def contruct_features(input_data):
    df = pd.DataFrame(input_data)

    df['text'] = df.groupby("map_id")['idea_title'].transform(
        lambda x: ' [SEP] '.join(x)).drop_duplicates()
    df_input = df.drop_duplicates("map_id").dropna(subset="text")
    df_input = df_input[~df_input['map_category_name'].isin(["Life", "Productivity", "Entertainment"])]

    df_input['text'] = "[CLS] " + df_input['map_title'] + " [SEP] " + df_input['text'] + " [SEP]"

    feature = df_input['text'][0]

    return feature


app = Flask(__name__)

MODEL = ClassificationModel(
    "bert", MODEL_CHECPOINT_PATH,
    use_cuda=False
)
LABELS_DICT = {0: 'Other', 1: 'Business', 2: 'Education', 3: 'Technology'}

@app.route("/predict", methods=['POST'])
def predict_category():
    input_data = request.json

    validate_input(input_data)
    title_feature = contruct_features(input_data)

    res = MODEL.predict([title_feature])[0][0]
    return {"predicted_category":LABELS_DICT.get(res)}

if __name__ == "__main__":
    app.run()