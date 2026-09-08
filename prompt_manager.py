"""
prompt_manager.py
Load and cache prompt templates.
"""

import os


class PromptManager:

    def __init__(self, base_dir="./prompts", version="v1"):
        self.prompt_dir = os.path.join(base_dir, version)
        self.cache = {}

    def get(self, name: str) -> str:

        if name in self.cache:
            return self.cache[name]

        path = os.path.join(self.prompt_dir, f"{name}")

        with open(path, "r", encoding="utf-8") as f:
            prompt = f.read()

        self.cache[name] = prompt
        return prompt


# 全局单例
prompt_manager = PromptManager()
