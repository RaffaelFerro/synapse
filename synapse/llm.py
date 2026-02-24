import json
import os
import urllib.request
from typing import Any, Optional

class LLMClient:
    """
    Cliente minimalista para comunicação com LLM usando apenas a biblioteca padrão.
    Suporta provedores compatíveis com a API do OpenAI ou Gemini (via adaptador).
    """
    def __init__(self):
        self.api_key = os.environ.get("SYNAPSE_LLM_KEY")
        self.endpoint = os.environ.get("SYNAPSE_LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions")
        self.model = os.environ.get("SYNAPSE_LLM_MODEL", "gpt-4-turbo-preview")

    def chat_completion(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        if not self.api_key:
            raise RuntimeError("SYNAPSE_LLM_KEY não configurada nas variáveis de ambiente.")

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"} if "json" in system_prompt.lower() else None,
            "temperature": 0.1
        }

        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(data).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Erro na chamada da LLM: {e}")
            return None

def get_llm_client() -> LLMClient:
    return LLMClient()
