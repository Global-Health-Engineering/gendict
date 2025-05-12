import requests
import json

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
