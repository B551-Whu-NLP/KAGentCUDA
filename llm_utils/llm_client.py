import os
from typing import List, Dict, Any, Optional, Tuple, Generator
from openai import OpenAI

from .logger import DialogLogger
from .cost_calculator import calculate_cost

ORG_CONFIG_MAP = {
    "openai": {
        "base_url_env": "OPENAI_BASE_URL",
        "api_key_env": "OPENAI_API_KEY",
        "default_base_url": "https://api.openai.com/v1/"
    },
    "ali": {
        "base_url_env": "ALI_BASE_URL",
        "api_key_env": "ALI_API_KEY",
        "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/"
    },
    "dp":{
        "base_url_env": "DP_BASE_URL",
        "api_key_env": "DP_API_KEY",
        "default_base_url": "https://api.deepseek.com"
    }
}


class OpenAILLM:
    def __init__(
            self,
            model_name: str,
            org: Optional[str] = None,
            base_url: Optional[str] = None,
            api_key: Optional[str] = None,
            log_path: str = "./logs",
            timeout: int = 60,
            max_retries: int = 3,
            temperature: float = 0.7,
            top_p: float = 0.9,
            verbose: bool = False,
    ):
        self.model_name = model_name
        self.org = org
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.top_p = top_p
        self.verbose = verbose

        self.base_url, self.api_key = self._resolve_config(org, base_url, api_key)

        if self.verbose:
            print(f"[DEBUG] Using org: {org}, base_url: {self.base_url}")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries
        )

        self.logger = DialogLogger(log_path=log_path, model_name=model_name, verbose=verbose)

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    def _resolve_config(self, org: Optional[str], base_url: Optional[str], api_key: Optional[str]) -> Tuple[str, Optional[str]]:
        resolved_base_url = base_url
        resolved_api_key = api_key

        if org:
            org = org.lower()
            if org in ORG_CONFIG_MAP:
                config = ORG_CONFIG_MAP[org]
                if not resolved_base_url:
                    resolved_base_url = os.getenv(config["base_url_env"], config["default_base_url"])
                if not resolved_api_key:
                    resolved_api_key = os.getenv(config["api_key_env"])
            else:
                if self.verbose:
                    print(f"[WARN] Unknown org: {org}, using defaults")
                if not resolved_base_url:
                    resolved_base_url = "https://api.openai.com/v1/"

        if not resolved_base_url:
            resolved_base_url = "https://api.openai.com/v1/"

        return resolved_base_url, resolved_api_key

    def chat(
            self,
            messages: List[Dict[str, str]],
            stream: bool = False,
            data_id: Optional[str] = None,
            turn_id: Optional[int] = None,
            **kwargs
    ) -> Tuple[str, int, int, float]:
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p) if "claude" not in self.model_name else None,
            "stream": stream,
            "extra_body": {"thinking": {"type": "disabled"}} if self.org=="dp" else {},
            **{k: v for k, v in kwargs.items() if k not in ["temperature", "top_p", "stream"]}
        }

        if self.verbose:
            print(f"[DEBUG] Calling model: {self.model_name}, stream: {stream}")
            print(f"[DEBUG] Messages: {messages}")

        if stream:
            content, input_tokens, output_tokens = self._stream_chat(params, messages)
        else:
            content, input_tokens, output_tokens = self._non_stream_chat(params)

        cost = calculate_cost(self.model_name, input_tokens, output_tokens)

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost

        system_prompt = self._extract_system_prompt(messages)
        last_user_prompt = self._extract_last_user_prompt(messages)

        self.logger.log_request(
            data_id=data_id,
            turn_id=turn_id,
            prompt=last_user_prompt,
            system_prompt=system_prompt,
            parameters={"temperature": params["temperature"], "top_p": params["top_p"]},
            response={"content": content},
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

        if self.verbose:
            print(f"[DEBUG] Response: {content[:100]}...")
            print(f"[DEBUG] Tokens: input={input_tokens}, output={output_tokens}, cost=${cost:.6f}")

        return content, input_tokens, output_tokens, cost

    def chat_stream(
            self,
            messages: List[Dict[str, str]],
            data_id: Optional[str] = None,
            turn_id: Optional[int] = None,
            **kwargs
    ) -> Generator[Tuple[str, bool], None, None]:
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "stream": True,
            "extra_body": {"thinking": {"type": "disabled"}}
            **{k: v for k, v in kwargs.items() if k not in ["temperature", "top_p", "stream"]}
        }

        if self.verbose:
            print(f"[DEBUG] Streaming model: {self.model_name}")

        response = self.client.chat.completions.create(**params)

        full_content = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_content += content
                yield content, False

        yield full_content, True

        input_tokens = 0
        output_tokens = len(full_content) // 4

        cost = calculate_cost(self.model_name, input_tokens, output_tokens)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost

        system_prompt = self._extract_system_prompt(messages)
        last_user_prompt = self._extract_last_user_prompt(messages)

        self.logger.log_request(
            data_id=data_id,
            turn_id=turn_id,
            prompt=last_user_prompt,
            system_prompt=system_prompt,
            parameters={"temperature": params["temperature"], "top_p": params["top_p"]},
            response={"content": full_content},
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

    def _non_stream_chat(self, params: dict) -> Tuple[str, int, int]:
        response = self.client.chat.completions.create(**params)

        content = response.choices[0].message.content
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens

        return content, input_tokens, output_tokens

    def _stream_chat(self, params: dict, messages: List[Dict[str, str]]) -> Tuple[str, int, int]:
        response = self.client.chat.completions.create(**params)

        full_content = ""
        input_tokens = 0
        output_tokens = 0

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens

        if output_tokens == 0:
            output_tokens = len(full_content) // 4

        return full_content, input_tokens, output_tokens

    def _extract_system_prompt(self, messages: List[Dict[str, str]]) -> str:
        for msg in messages:
            if msg.get("role") == "system":
                return msg.get("content", "")
        return ""

    def _extract_last_user_prompt(self, messages: List[Dict[str, str]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "org": self.org
        }

    def reset_stats(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
