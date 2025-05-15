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

path_to_data = ""path/to/data.csv""
response, descriptions = core.gendict("YOUR_HELMHOLTZ_API_KEY",
                                      path_to_data, 
                                      max_unique_values=7, 
                                      model=1, 
                                      temperature=0, 
                                      top_p=0.5, 
                                      max_tokens=999, 
                                      return_response=True)
```