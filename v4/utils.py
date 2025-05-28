import json


def load_configuration(config_path):
    with open(config_path) as file:
        return json.load(file)
