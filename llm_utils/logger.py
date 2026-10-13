import os
import json
from datetime import datetime


class DialogLogger:
    def __init__(self, log_path: str, model_name: str, verbose: bool = False):
        self.verbose = verbose
        timestamp = datetime.now().strftime("%Y%m%d%H")
        self.filename = f"{model_name}_{timestamp}.jsonl"
        self.filepath = os.path.join(log_path, self.filename)
        os.makedirs(log_path, exist_ok=True)

    def log_request(self, data_id, turn_id, prompt: str, system_prompt: str,
                    parameters: dict, response: dict,
                    input_tokens: int, output_tokens: int):
        log_entry = {
            "data_id": data_id,
            "turn_id": turn_id,
            "timestamp": datetime.now().isoformat(),
            "model": self.filename.split("_")[0],
            "prompt": prompt,
            "system_prompt": system_prompt,
            "parameters": parameters,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "response": response
        }

        if self.verbose:
            print(f"[DEBUG] Logging request:\n{json.dumps(log_entry, indent=2, ensure_ascii=False)}")

        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
