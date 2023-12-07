import json
import pandas as pd

# ----------------------------------------------------
path = r"..\data\train_output.json"  # change the path for each file
json_data = []

with open(path, "r", encoding="utf-8") as file:
    for line in file:
        try:
            json_obj = json.loads(line)
            json_data.append(json_obj)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            continue  # continues if it has an error

print(f"The number of data loaded: {len(json_data)}")


# ----------------------------------------------------
# calculate the average rating
def calculate_average_rating(json_data):
    # Mapping ratings to numeric values
    rating_values = {"Excellent": 4, "Acceptable": 3, "Could be Improved": 2, "Bad": 1}

    # Dictionary to store the sum of ratings and count for each answer
    answer_ratings = {}

    # Process each entry in the JSON data
    for entry in json_data:
        answer = entry["Answer"]
        rating = entry["Rating"]

        # Initialize the answer in the dictionary if not present
        if answer not in answer_ratings:
            answer_ratings[answer] = {"total": 0, "count": 0}

        # Add the numeric value of the rating to the total and increment count
        answer_ratings[answer]["total"] += rating_values[rating]
        answer_ratings[answer]["count"] += 1

    # Calculate the average rating for each answer
    average_ratings = {}
    for answer, data in answer_ratings.items():
        average_ratings[answer] = data["total"] / data["count"]

    return average_ratings


average_ratings = calculate_average_rating(json_data)

# ----------------------------------------------------
formatted_output_with_scores = []
rating_values = {"Excellent": 4, "Acceptable": 3, "Could be Improved": 2, "Bad": 1}

for entry in json_data:
    question = entry["Question"]
    answer = entry["Answer"]
    rating = entry["Rating"]
    score = rating_values.get(rating, 0)  # Get the numeric score for the rating
    average = average_ratings[answer]
    formatted_entry = {
        "Question": question,
        "Answer": answer,
        "Rating": rating,
        "Score": score,
        "Average": average,
    }
    formatted_output_with_scores.append(formatted_entry)

# ----------------------------------------------------
df = pd.DataFrame(formatted_output_with_scores)

# ----------------------------------------------------
json_filename = "../data/train_formatted_output.json"
df.to_json(json_filename, orient="records", lines=True)
