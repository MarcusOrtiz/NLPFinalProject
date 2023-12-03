import pandas as pd
import json

# Extract json files ----------------------------------------
path = "../../data/feedback_test.json"
valid = json.load(open(path))

with open(path, "r") as file:
    valid = json.load(file)

for item in valid:
    item["passage"]["reference"]["section_content"] = item["passage"]["reference"][
        "section_content"
    ].replace("\n", " ")

# ------------------------------------------------------------
records = []
for item in valid:
    question = item["question"]
    section_content = item["passage"]["reference"]["section_content"]
    for rating in item["rating"]:
        records.append(
            {"Question": question, "Section_Content": section_content, "Rating": rating}
        )

df = pd.DataFrame(records)
df.rename(columns={"Section_Content": "Answer"}, inplace=True)

# Save the output files
# ------------------------------------------------------------
output_file_path = "../data/output.json"
df.to_json(output_file_path, orient="records", lines=True)

print(f"DataFrame has been saved as '{output_file_path}'.")
