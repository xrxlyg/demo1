# llm_wrapper.py
import os
import json
from typing import Any, Dict, Optional
from openai import OpenAI


class LLMWrapper:
    """
    使用千问3（DashScope OpenAI-compatible API）
    保持 generate / generate_json 行为不变
    """

    def __init__(
        self,
        model_name: str = "qwen-plus",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = OpenAI(
            api_key="sk-928d1063dc7d43e0a161cd8fcd2cef27",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        **kwargs,
    ) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return resp.choices[0].message.content.strip()

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        schema: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if schema is not None:
            prompt += (
                "\n请严格按照以下 JSON Schema 输出，只输出 JSON：\n"
                f"{json.dumps(schema, ensure_ascii=False)}"
            )
        else:
            prompt += "\n请只输出合法的 JSON，不要包含任何其他内容。"

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )

        return json.loads(resp.choices[0].message.content)


