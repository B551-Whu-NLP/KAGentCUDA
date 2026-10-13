COSTS = {
    "input": {
        "gpt-3.5-turbo": 0.006,
        "gpt-4.1": 0.024,
        "gpt-4o": 0.03,
        "gpt-4o-mini": 0.0018,
        "o1-mini": 0.036,
        "o3": 0.12,
        "o4-mini": 0.0132,
        "gpt-4": 0.36,
        "claude-3-sonnet-20240620": 0.036,
        "ERNIE-4.5-turbo-32K": 0.0008,
        "ERNIE-4.5-turbo-128K": 0.0008,
        "deepseek-v3": 0.002,
        "deepseek-r1": 0.004,
        "Qianfan-8b": 0.0005,
        "qwq-plus": 0.0016,
        "qwen-max": 0.0024,
        "qwen-plus": 0.0008,
        "qwen-turbo": 0.0003,
        "qwen-coder-plus": 0.0035,
        "qwen-coder-turbo": 0.002,
        "qwen3-32b": 0.002,
        "qwen3-coder-next-32k": 0.0014,
        "qwen3-coder-next-128k": 0.0021,
        "qwen3-coder-next-256k": 0.0036
    },
    "output": {
        "gpt-3.5-turbo": 0.018,
        "gpt-4.1": 0.096,
        "gpt-4o": 0.12,
        "gpt-4o-mini": 0.0072,
        "o1-mini": 0.144,
        "o3": 0.48,
        "o4-mini": 0.0528,
        "gpt-4": 0.72,
        "claude-3-sonnet-20240620": 0.18,
        "ERNIE-4.5-turbo-32K": 0.0032,
        "ERNIE-4.5-turbo-128K": 0.0032,
        "deepseek-v3": 0.008,
        "deepseek-r1": 0.016,
        "Qianfan-8b": 0.001,
        "qwq-plus": 0.004,
        "qwen-max": 0.0096,
        "qwen-plus": 0.0002,
        "qwen-turbo": 0.0006,
        "qwen-coder-plus": 0.007,
        "qwen-coder-turbo": 0.006,
        "qwen3-32b": 0.02,
        "qwen3-coder-next-32k": 0.0057,
        "qwen3-coder-next-128k": 0.0086,
        "qwen3-coder-next-256k": 0.0143
    }
}

QWEN3_CODER_NEXT_TIERS = [
    {"max_tokens": 32768, "input_rate": 1.0, "output_rate": 4.0},
    {"max_tokens": 131072, "input_rate": 1.5, "output_rate": 6.0},
    {"max_tokens": 262144, "input_rate": 2.5, "output_rate": 10.0}
]

CNY_TO_USD = 0.14


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    # 检查是否是 qwen3-coder-next 系列（带后缀的版本）
    if model_name.startswith("qwen3-coder-next"):
        total_tokens = input_tokens + output_tokens
        tier = None
        for t in QWEN3_CODER_NEXT_TIERS:
            if total_tokens <= t["max_tokens"]:
                tier = t
                break
        
        if tier is None:
            tier = QWEN3_CODER_NEXT_TIERS[-1]
        
        input_cost_cny = (input_tokens / 1_000_000) * tier["input_rate"]
        output_cost_cny = (output_tokens / 1_000_000) * tier["output_rate"]
        return (input_cost_cny + output_cost_cny) * CNY_TO_USD
    
    if model_name not in COSTS['input'] or model_name not in COSTS['output']:
        return 0.0

    cost_per_input_token = COSTS['input'][model_name] / 1000
    cost_per_output_token = COSTS['output'][model_name] / 1000

    return (input_tokens * cost_per_input_token) + (output_tokens * cost_per_output_token)
