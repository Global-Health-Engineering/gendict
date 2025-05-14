# gendict

A package to generate tidy descscriptions for variables in a dataset.

## Installation

```{bash}
git add submodule https://github.com/Global-Health-Engineering/gendict.git
```

## Example

```{python}
import os
import json
from gendict.gendict import core

config_path = os.path.join(os.path.expanduser("~"), 'config.json')

with open(config_path, 'r') as file:
    config = json.load(file)

path_to_data = ""path/to/data.csv""
response, descriptions = core.gendict(config["HELMHOLTZ_API_KEY"],
                                      path_to_data, 
                                      max_unique_values=7, 
                                      model=1, 
                                      temperature=0, 
                                      top_p=0.5, 
                                      max_tokens=999, 
                                      return_response=True)
```