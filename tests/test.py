
import os
import json

from gendict.core import gendict, upload_descriptions_from_json

config_path = os.path.join(os.path.expanduser("~"), 'config.json')

with open(config_path, 'r') as file:
    config = json.load(file)

context, response, json_content = gendict(config["HELMHOLTZ_API_KEY"], "ethord.csv", max_unique_values=7, model=1, temperature=0, top_p=0.5, max_tokens=999, debug=True)
print(f"Response:\n{response}\n\nDescriptions:\n{json_content}")

if json_content:
    upload_descriptions_from_json(json_content, in_file="dictionary.csv")
