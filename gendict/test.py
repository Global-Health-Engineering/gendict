
import os
import json

from core import gendict, upload_descriptions_from_json

config_path = os.path.join(os.path.expanduser("~"), 'config.json')

with open(config_path, 'r') as file:
    config = json.load(file)

response, descriptions = gendict(config["HELMHOLTZ_API_KEY"], "../tests/ethord.csv", max_unique_values=7, model=1, temperature=0, top_p=0.5, max_tokens=999, return_response=True)

# save response to txt
with open('response.txt', 'w') as file:
    file.write(response)

if descriptions:
    print(descriptions)
    upload_descriptions_from_json(descriptions, in_file="../tests/dictionary.csv")
