import os
import json

import requests
import re
import pandas as pd


class Blablador():

    def __init__(self, api_key, model = None, temperature = 0.7, top_p = 1.0, choices = 1, max_tokens =  50, user = "default"):
        self.api_key = api_key

        self.url_root = "https://helmholtz-blablador.fz-juelich.de:8000/v1/"
        self.headers = {'accept': 'application/json', 'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}

        if model is None:
            self.model = self.models()[0]
        elif type(model) is int:
            try:
                self.model = self.models()[model]
            except IndexError:
                raise ValueError(f"Invalid model index: {model}. Available models are indexed from 0 to {len(self.models()) - 1}")
        elif isinstance(model, str):
            try:
                if model in self.models():
                    self.model = model
                else:
                    raise ValueError(f"Model '{model}' not found in available models: {self.models()}")
            except Exception as e:
                print(f"An unexpected error occurred while checking the model: {e}")
                raise
        else:
            raise TypeError(f"Invalid model type: {type(model)}. Expected None, int, or str.")

        print(f"Selected model: {self.model}")

        self.temperature = temperature
        self.top_p = top_p
        self.choices = choices
        self.max_tokens = max_tokens
        self.user = user
        
        # Default values from https://helmholtz-blablador.fz-juelich.de:8000/docs#/
        self.suffix = "string"
        self.logprobs = 0
        self.echo = "false"
        self.presence_penalty = 0
        self.frequency_penalty = 0
    
    def models(self, verbose=False):
        url = self.url_root+"models"
        response = requests.get(url = url, headers = self.headers)
        models = json.loads(response.text)["data"]

        # TODO write error messages for 400, 401, etc respones
        # like with response.ok , response.status, etc... 
        if verbose:
            return models
        else:
            ids = []
            for model in models:
                ids.append(model["id"])
            return(ids)
    
    def completion(self, prompt, verbose=False):
        url = self.url_root+"completions"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "suffix": self.suffix,
            "temperature": self.temperature,
            "n": self.choices,
            "max_tokens": self.max_tokens,
            "stop": [
                "string"
            ],
            "stream": "false",
            "top_p": self.top_p,
            "logprobs":self.logprobs,
            "echo":self.echo,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "user": self.user
        }

        payload = json.dumps(payload)
        
        response = requests.post(url = url, headers = self.headers, data=payload)
        # TODO write error messages for 400, 401, etc respones
        # like with response.ok , response.status, etc... 
        if verbose:
            return(response)
        else:
            if self.choices == 1:
                return(json.loads(response.text).get("choices")[0].get("text"))
            else:
                return(json.loads(response.text).get("choices"))


def get_limited_unique_values(in_file, max_unique_values=None):
    """
    Reads a CSV file and returns a dictionary of unique values from each column,
    optionally limited to a specified number.

    Args:
        in_file (str): The path to the CSV file.
        max_unique_values (int, optional): The maximum number of unique values to return
                                for each column. Defaults to None (all unique values).

    Returns:
        dict: A dictionary where keys are column names and values are lists
              of unique values. Returns None if the file cannot be read.
    """
    try:
        data_in = pd.read_csv(in_file)
    except FileNotFoundError:
        print(f"Error: File not found at {in_file}")
        return None
    except Exception as e:
        print(f"Error reading file {in_file}: {e}")
        return None

    unique_dict = {}
    for col in data_in.columns:
        unique_values = data_in[col].unique().tolist()
        if max_unique_values is not None and len(unique_values) > max_unique_values:
            unique_dict[col] = unique_values[:max_unique_values]
        else:
            unique_dict[col] = unique_values

    return unique_dict

def extract_json_from_llm_output(llm_output):
    """
    Extracts JSON content from an LLM output string, handling various code block formats.

    Args:
        llm_output (str): The string output from the LLM.

    Returns:
        dict or None: A dictionary representing the extracted JSON, or None if no valid JSON is found.
    """
    # Try to find JSON within a JSON code block (```json ... ```)
    json_match = re.search(r"```(?:json)?(.*?)```", llm_output, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON within JSON code block: {e}")

    # Try to parse the entire output as JSON (in case it's not in a code block)
    try:
        return json.loads(llm_output.strip())
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from the entire output: {e}")

    return None

def write_descriptions_to_csv(json_data, out_file="dictionary.csv"):
    """
    Writes variable descriptions from a JSON dictionary to a CSV file.

    Args:
        json_data (dict): A dictionary where keys are variable names and values are their descriptions.
        out_file (str, optional): The path to the CSV file. Defaults to "dictionary.csv".
    """
    try:
        # Check if the file exists.  If it does, we'll append to it.
        file_exists = os.path.exists(out_file)

        # Create a Pandas DataFrame from the JSON data
        data = {'variable_name': list(json_data.keys()), 'description': list(json_data.values())}
        df = pd.DataFrame(data)

        # Write the dataframe to CSV.  Include the header only if the file didn't exist.
        df.to_csv(out_file, mode='a', header=not file_exists, index=False, encoding='utf-8')

        print(f"Descriptions successfully written to {out_file}")

    except Exception as e:
        print(f"Error writing to CSV file: {e}")

def upload_descriptions_from_json(json_data, in_file):
    """
    Updates descriptions in an existing CSV file from a JSON dictionary.

    Args:
        json_data (dict): A dictionary where keys are variable names and values are their descriptions.
        in_file (str, optional): The path to the CSV file to update. Defaults to "dictionary.csv".
    """
    try:
        # Check if the input file exists
        if not os.path.exists(in_file):
            raise FileNotFoundError(f"Error: Input file '{in_file}' not found.")

        # Read the existing CSV file into a Pandas DataFrame
        df = pd.read_csv(in_file, encoding='utf-8')

        # Check if the 'description' column exists
        if 'description' not in df.columns:
            raise ValueError(f"Error: The CSV file '{in_file}' does not contain a 'description' column.")

        # Create a Series from the JSON data for easy lookup
        new_descriptions = pd.Series(json_data)

        # Identify rows in the CSV where the 'variable_name' (assuming it exists as index or column)
        # matches the keys in the json_data. Update the 'description' for those rows.
        # We'll first check if 'variable_name' is a column, if not, we'll assume the index.
        if 'variable_name' in df.columns:
            df['description'] = df['variable_name'].map(new_descriptions).fillna(df['description'])
        elif df.index.name is not None:
            df['description'] = df.index.map(new_descriptions).fillna(df['description'])
        else:
            raise ValueError(f"Error: The CSV file '{in_file}' must have either a 'variable_name' column or an index matching the keys in the JSON data.")

        # Write the updated DataFrame back to the CSV file, overwriting the original
        df.to_csv(in_file, index=True if df.index.name is not None else False, encoding='utf-8')

        print(f"Descriptions in '{in_file}' successfully updated.")

    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def get_context(in_file, max_unique_values=7):
    limited_unique_values = get_limited_unique_values(in_file, max_unique_values=max_unique_values)

    context = """
    You will receive a Python dictionary where the keys are the names of columns from a CSV file, and the values are lists containing up to `max_unique_values`=%d unique example values from those columns.

    Your primary goal is to generate clear, concise, and informative data dictionary entries for each column. These descriptions will be used to help individuals unfamiliar with the dataset understand the meaning and type of information each column contains for documentation purposes. Aim for descriptions that are no more than two sentences long.

    Consider the following steps for each column:

    1.  **Analyze the column name:** What does the name itself suggest about the data it might contain?
    2.  **Examine the provided example values:** Look for patterns, units, categories, or ranges that can help you understand the nature of the data.
    3.  **Infer the likely meaning and context:** Based on the column name and example values, what real-world concept or measurement does this column likely represent? Try to infer the broader domain or field of study this data might belong to (e.g., environmental science, social surveys, medical records).
    4.  **Determine the data type:** Clearly identify the likely data type of the column (e.g., categorical, continuous numerical, discrete numerical, text, boolean, date).
    5.  **Write a concise description:** Combine your inferences into a brief description (1-2 sentences) that explains the meaning of the column and its data type. Use clear and accessible language, avoiding overly technical jargon unless essential.

    For example, if you receive the following dictionary (with a `max_unique_values`=7):

    {'species': ['Adelie', 'Gentoo', 'Chinstrap'],
    'island': ['Torgersen', 'Biscoe', 'Dream'],
    'bill_length_mm': [39.1, 39.5, 40.3, nan, 36.7, 39.3, 38.9],
    'bill_depth_mm': [18.7, 17.4, 18.0, nan, 19.3, 20.6, 17.8],
    'flipper_length_mm': [181.0, 186.0, 195.0, nan, 193.0, 190.0, 180.0],
    'body_mass_g': [3750.0, 3800.0, 3250.0, nan, 3450.0, 3650.0, 3625.0],
    'sex': ['male', 'female', nan],
    'year': [2007, 2008, 2009]}

    You should aim to provide a JSON that looks like this:

    {
    "species": "The species of penguin observed, with categories including Adelie, Chinstrap, and Gentoo.",
    "island": "The specific island where the penguin was observed, such as Torgersen, Biscoe, or Dream.",
    "bill_length_mm": "The length of the penguin's bill (beak) measured in millimeters. This is a continuous numerical measurement.",
    "bill_depth_mm": "The depth of the penguin's bill measured in millimeters. This is a continuous numerical measurement.",
    "flipper_length_mm": "The length of the penguin's flipper measured in millimeters. This is a discrete numerical measurement.",
    "body_mass_g": "The body mass of the penguin recorded in grams. This is a discrete numerical measurement.",
    "sex": "The biological sex of the penguin, categorized as either male or female.",
    "year": "The year in which the penguin observation was recorded. This is a discrete numerical value representing the year of data collection."
    }

    The dictionary you will process is:

    %s

    Your response must be in JSON format, containing a dictionary where each key is a column name from the input dictionary, and the value is the generated description for that column. 
    Output *only* this JSON dictionary.
    """ % (max_unique_values, limited_unique_values)
    return context

def gendict(API_KEY, in_file, max_unique_values=7, model=1, temperature=0, top_p=0.5, max_tokens=999, debug=False):
    # Config Blablador
    blablador = Blablador(API_KEY, model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    # Context
    context = get_context(in_file, max_unique_values=max_unique_values)

    # Generate descriptions.json
    response = blablador.completion(context)

    json_content = extract_json_from_llm_output(response)

    if debug:
        return context, response, json_content
    else:
        return json_content