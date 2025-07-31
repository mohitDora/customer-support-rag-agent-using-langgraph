import json
import os

from langchain.prompts.chat import ChatPromptTemplate


def read_json(file_path):
    try:
        with open(file_path, "r") as f:
            content = f.read().strip()
            return json.loads(content) if content else {}
    except FileNotFoundError:
        return {}


def save_json(data, file_path):
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
    except FileNotFoundError:
        print(f"Error saving data to {file_path}: File not found")
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")


def create_directories(path_to_directories: list, verbose=True):
    for path in path_to_directories:
        if os.path.exists(path):
            if verbose:
                print(f"directory already exists at: {path}")
            continue
        os.makedirs(path, exist_ok=True)
        if verbose:
            print(f"created directory at: {path}")


def read_txt(file_path):
    with open(file_path, "r") as f:
        return f.read().strip()


def return_prompt_template(file_path) -> ChatPromptTemplate:
    with open(file_path, "r", encoding="utf-8") as f:
        template = f.read().strip()
    return ChatPromptTemplate.from_messages([("system", template)])
