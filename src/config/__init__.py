import os
from dotenv import load_dotenv
from src.utils.common import read_json
from src.constants import BASE_DIR
from pathlib import Path

load_dotenv()


class ConfigurationManager:
    def __init__(self):
        self.config = read_json(Path.joinpath(BASE_DIR, "config/config.json"))

    def get_knowledge_base_config(self):
        config = self.config["knowledge_base"]
        return config

    def get_llm_config(self):
        config = self.config["llm_config"]
        config["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
        return config

    def get_agent_config(self):
        config = self.config["agent_config"]
        return config

if __name__ == "__main__":
    config = ConfigurationManager()
    knowledge_base_config = config.get_knowledge_base_config()
    print(knowledge_base_config)