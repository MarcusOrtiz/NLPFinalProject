from .load_datasets import open_json
import json
from pathlib import Path

TRAIN_DATA_PATH = 'train_formatted_output_w_comma.json'
VAL_DATA_PATH = 'valid_formatted_output_w_comma.json'
TEST_DATA_PATH = 'test_formatted_output_w_comma.json'

NEW_TRAIN_DATA_PATH = Path('./data/unique_answers/train_data_classification.json').resolve()
NEW_VAL_DATA_PATH = Path('./data/unique_answers/val_data_classification.json').resolve()
NEW_TEST_DATA_PATH = Path('./data/unique_answers/test_data_classification.json').resolve()

for path in [TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH]:
    data = open_json(path)

    unique_answers = set()
    filtered_json_array = []

    for pair in data:
        answer = pair.get("Answer")
        average = pair.get("Average")

        if answer not in unique_answers:
            unique_answers.add(answer)
            # Categorize the average based on specified ranges
            if 1 <= average <= 1.75:
                pair["Average"] = 1
            elif 1.75 < average <= 2.5:
                pair["Average"] = 2
            elif 2.5 < average <= 3.25:
                pair["Average"] = 3
            elif 3.25 < average <= 4:
                pair["Average"] = 4

            filtered_json_array.append(pair)

    print(f'path: {path}')
    if path == TRAIN_DATA_PATH:
        print('train file saving')
        with open(NEW_TRAIN_DATA_PATH, 'w') as file:
            json.dump(filtered_json_array, file)
    elif path == VAL_DATA_PATH:
        print('val file saving')
        with open(NEW_VAL_DATA_PATH, 'w') as file:
            json.dump(filtered_json_array, file)
    elif path == TEST_DATA_PATH:
        with open(NEW_TEST_DATA_PATH, 'w') as file:
            json.dump(filtered_json_array, file)