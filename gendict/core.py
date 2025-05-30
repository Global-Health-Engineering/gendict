import os
import json

import re
import pandas as pd
import numpy as np

from blablador.core import Blablador


def get_limited_unique_values(in_file, max_unique_values=None, max_value_length=None):
    """
    Reads a CSV file and returns a dictionary of unique values from each column,
    optionally limited to a specified number and length.

    Args:
        in_file (str): The path to the CSV file.
        max_unique_values (int, optional): The maximum number of unique values to return
                                for each column. Defaults to None (all unique values).
        max_value_length (int, optional): The maximum length of each unique value. Defaults to None (no length limit).

    Returns:
        dict: A dictionary where keys are column names and values are lists
              of unique values. Returns None if the file cannot be read.
    """
    try:
        # Attempt to read the CSV file
        data_in = pd.read_csv(in_file)
    except FileNotFoundError:
        # Handle the case where the file is not found
        print(f"Error: File not found at {in_file}")
        return None
    except Exception as e:
        # Handle any other exceptions that occur while reading the file
        print(f"Error reading file {in_file}: {e}")
        return None

    # Initialize an empty dictionary to store unique values
    unique_dict = {}
    for col in data_in.columns:
        # Get the unique values for the current column
        unique_values = data_in[col].unique().tolist()
        
        # Truncate unique values based on the maximum length
        if max_value_length is not None:
            unique_values = [str(val)[:max_value_length] for val in unique_values]
        
        # Check if a limit is specified and if the number of unique values exceeds this limit
        if max_unique_values is not None and len(unique_values) > max_unique_values:
            # If a limit is specified and exceeded, truncate the list of unique values
            unique_dict[col] = unique_values[:max_unique_values]
        else:
            # If no limit is specified or the number of unique values does not exceed the limit, store all unique values
            unique_dict[col] = unique_values

    # Return the dictionary of unique values
    return unique_dict

def extract_json_from_llm_output(llm_output_string):
    """
    Extracts a JSON string from an LLM's raw text output,
    handling common issues like special tokens, markdown code blocks,
    and leading/trailing whitespace.
    """
    # 1. Remove common special tokens (specific to some LLMs like Llama variants)
    #    You might need to adjust this regex based on the exact tokens your model emits.
    cleaned_string = re.sub(r'<\|.*?\|>|\[INST\]|\[/INST\]', '', llm_output_string)
    
    # 2. Remove markdown code block delimiters if present
    #    This regex matches ``` optionally followed by "json" or other language identifier,
    #    and captures the content within.
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', cleaned_string, re.DOTALL)
    if match:
        json_string = match.group(1).strip()
    else:
        # If no markdown block is found, assume the entire cleaned string is the JSON
        json_string = cleaned_string.strip()
    
    # 3. Attempt to find the first '{' and last '}' to isolate the JSON object
    #    This helps if there's leading/trailing non-JSON text the regex missed.
    first_brace = json_string.find('{')
    last_brace = json_string.rfind('}')
    
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_string = json_string[first_brace : last_brace + 1]
    
    # 4. Attempt to parse the JSON
    try:
        # Validate that it's parsable JSON
        json.loads(json_string) 
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse cleaned JSON string. Error: {e}")
        print(f"Attempted JSON string:\n{json_string}")
        # Fallback for when extraction fails, perhaps return an empty dict or raise error
        # In a real scenario, you might want to log this or return a special value
        return "" # Return an empty string if parsing fails after all attempts

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

def upload_descriptions_from_json(in_file, json_data, dictionary_file):
    """
    Updates descriptions in an existing CSV file from a JSON dictionary.
    The update is performed by matching both the variable name and the file name.

    Args:
        in_file (str): The path to the dataframe the json_data refers to (e.g., "inst/ext/details.csv").
                       The base name of this file (without path or extension) is used for matching.
        json_data (dict): A dictionary where keys are variable names and values are their descriptions.
        dictionary_file (str): The path to the CSV file to update.
    """
    try:
        # Check if the input dictionary file exists
        if not os.path.exists(dictionary_file):
            raise FileNotFoundError(f"Error: Input dictionary file '{dictionary_file}' not found.")

        # Read the existing CSV file into a Pandas DataFrame
        df = pd.read_csv(dictionary_file, encoding='utf-8')

        # Check if all necessary columns exist in the DataFrame
        required_columns = ['description', 'variable_name', 'file_name']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Error: The CSV file '{dictionary_file}' does not contain a '{col}' column.")

        base_in_file_name = os.path.splitext(os.path.basename(in_file))[0]

        # Iterate through the provided JSON data to update descriptions
        for var_name, description in json_data.items():
            # Create a boolean mask to identify rows that match both:
            # 1. The file name (after stripping '.csv' from the 'file_name' column)
            # 2. The variable name
            # Using regex=False for str.replace ensures a literal string replacement.
            mask = (df['file_name'].str.replace('.rda', '', regex=False) == base_in_file_name) & \
                   (df['variable_name'] == var_name)

            # Apply the update to the 'description' column for the rows identified by the mask.
            # .loc is used for label-based indexing to ensure direct modification of the DataFrame.
            df.loc[mask, 'description'] = description

        # Write the updated DataFrame back to the CSV file, overwriting the original.
        # index=False prevents Pandas from writing the DataFrame index as a column in the CSV.
        df.to_csv(dictionary_file, index=False, encoding='utf-8')

        print(f"Descriptions in '{dictionary_file}' successfully updated for variables in '{in_file}'.")

    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


import json # Make sure json is imported
import math # To handle nan correctly in json.dumps, if nan is a float type. For true NaN, json.dumps might need a custom encoder or pre-processing.
from math import nan # Assuming nan comes from math, or numpy if you're using pandas/numpy

def get_context(in_file, max_unique_values, max_value_length, general_context):
    penguins_unique_dict = {'species': ['Adelie', 'Gentoo', 'Chinstrap'],
                        'island': ['Torgersen', 'Biscoe', 'Dream'],
                        'bill_length_mm': [39.1, 39.5, 40.3, nan, 36.7, 39.3, 38.9],
                        'bill_depth_mm': [18.7, 17.4, 18.0, nan, 19.3, 20.6, 17.8],
                        'flipper_length_mm': [181.0, 186.0, 195.0, nan, 193.0, 190.0, 180.0],
                        'body_mass_g': [3750.0, 3800.0, 3250.0, nan, 3450.0, 3650.0, 3625.0],
                        'sex': ['male', 'female', nan],
                        'year': [2007, 2008, 2009]}
    penguins_result_dict = {"species": "The species of penguin observed, with categories including Adelie, Chinstrap, and Gentoo.",
                            "island": "The specific island where the penguin was observed, such as Torgersen, Biscoe, or Dream.",
                            "bill_length_mm": "The length of the penguin's bill (beak) measured in millimeters. This is a continuous numerical measurement.",
                            "bill_depth_mm": "The depth of the penguin's bill measured in millimeters. This is a continuous numerical measurement.",
                            "flipper_length_mm": "The length of the penguin's flipper measured in millimeters. This is a discrete numerical measurement.",
                            "body_mass_g": "The body mass of the penguin recorded in grams. This is a discrete numerical measurement.",
                            "sex": "The biological sex of the penguin, categorized as either male or female.",
                            "year": "The year in which the penguin observation was recorded. This is a discrete numerical value representing the year of data collection."
                            }

    unique_dict = get_limited_unique_values(in_file, max_unique_values=max_unique_values, max_value_length=max_value_length)

    # Pre-process unique_dict to handle nan if it's a float, convert to None for JSON
    def convert_nan_to_string(obj):
        if isinstance(obj, dict):
            return {k: convert_nan_to_string(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_nan_to_string(elem) for elem in obj]
        elif isinstance(obj, float) and math.isnan(obj):
            return "NaN" # Convert to string "NaN"
        return obj

    processed_unique_dict = convert_nan_to_string(unique_dict)
    processed_penguins_unique_dict = convert_nan_to_string(penguins_unique_dict)

    keys_dict = {key: "" for key in unique_dict} # Initialize with empty strings for consistency

    context = f"""You are an expert data documenter. Your task is to generate a precise codebook based on provided column samples.
                  You will receive a Python dictionary where the keys are the names of columns from a CSV file, and the values are lists containing up to `{max_unique_values}` unique example values from those columns. These examples include 'null' or 'None' values where data might be missing.
                  Your primary goal is to generate clear, concise, and informative documentation (a codebook) for each column. These descriptions will be used to help individuals unfamiliar with the dataset understand the meaning and type of information each column contains for documentation purposes. Aim for descriptions that are no more than two sentences long and under 30 words, if possible, while remaining informative. Do not invent information or make assumptions beyond what is directly supported by the column name and the provided example values.
                  
                  **IMPORTANT:**
                  * Descriptions must be definitive statements about the column's content and purpose.
                  * Do not speculate or use phrases like 'likely', 'possibly', 'may be', 'but no values are provided', 'no data available', or similar uncertain/apologetic language.
                  * Describe what the column *is* or *represents*, not what data *is missing* from the examples. If examples are 'null', describe the *expected* data type and meaning of the column as if data *were* present.
                  * Do not include any conversational filler, introductory remarks, or concluding statements.

                  General context:
                  {json.dumps(general_context, indent=2)}

                  Consider the following steps for each column:
                  
                  1.  **Analyze the column name:** What does the name itself suggest about the data it might contain?
                  2.  **Examine the provided example values:** Look for patterns, units, categories, or ranges that can help you understand the nature of the data. Pay attention to the presence of 'null' or 'None' values, indicating missing data.
                  3.  **Infer the likely meaning and context:** Based on the column name and example values, what real-world concept or measurement does this column likely represent? Try to infer the broader domain or field of study this data might belong to (e.g., environmental science, social surveys, medical records).
                  4.  **Determine the data type:** Clearly identify the likely data type of the column. Use precise terms like 'Categorical', 'Continuous Numerical', 'Discrete Numerical', 'Text', 'Boolean', 'Date', or 'Identifier'.
                  5.  **Write a concise description:** Combine your inferences into a brief description (1-2 sentences) that explains what the column *represents* in the real world and its inferred data type. Use clear and accessible language, avoiding overly technical jargon unless essential.
                  
                  For example, if you receive the following unique values dictionary (with a `max_unique_values`=7):\n {json.dumps(processed_penguins_unique_dict, indent=2)}
                  You should aim to provide a JSON that looks like this: \n {json.dumps(penguins_result_dict, indent=2)}
                  The unique values dictionary you have to process is: \n {json.dumps(processed_unique_dict, indent=2)}
                  
                  Your response must ONLY be the raw JSON object. Do not include any other text, introductory phrases, conversational elements, special tokens (e.g., <|eom_id|>, <|start_header_id|>), or markdown code block delimiters (e.g., ```json). Your response must begin directly with the opening curly brace of the JSON object.

                  Complete with descriptions this JSON: \n {json.dumps(keys_dict, indent=2)}
               """
    return context

def gendict(API_KEY, in_file, max_unique_values=7, max_value_length=100, general_context={}, model=1, temperature=0, top_p=0.5, max_tokens=999, debug=False):
    # Config Blablador
    blablador = Blablador(API_KEY, model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    # Context
    context = get_context(in_file, max_unique_values=max_unique_values, max_value_length=max_value_length, general_context=general_context)

    # Generate descriptions.json
    response = blablador.completion(context)

    json_content = extract_json_from_llm_output(response)

    if debug:
        return context, response, json_content
    else:
        return json_content