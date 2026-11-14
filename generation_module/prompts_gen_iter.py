"""
C → CUDA 迭代优化器的所有 prompt 模板集中在此。
保持纯函数 / 纯字符串，不依赖业务对象，便于单独迭代调优。
"""

from typing import List, Optional, Tuple


# ---------- System prompts ----------

SYSTEM_TRANSLATE_BASE = """You are an expert in CUDA parallel programming.
Your task is to translate the given sequential C program into a complete,
correct, and high-performance CUDA program."""


SYSTEM_CODE_ONLY_GENERATOR = """You are a CUDA code generator. Your ONLY output is the complete CUDA code.
禁止输出任何解释、分析或说明文字。只输出代码，不要任何其他内容。"""


SYSTEM_CODE_ONLY_OPTIMIZER = """You are a CUDA code optimizer. Your ONLY output is the complete optimized CUDA code.
禁止输出任何解释、分析或说明文字。只输出代码，不要任何其他内容。"""


# ---------- User prompts ----------

USER_TRANSLATE_INITIAL = """{reference_section}## Task
Translate the following C program into CUDA. The output must:
1. Implement the same computation as the input C program.
2. Use __global__ kernel functions with
threadIdx/blockIdx for parallelization.
3. Include all necessary host-side code:
memory allocation, host-device data transfers,
kernel invocation, and result retrieval.
4. Compile successfully with nvcc.
## Input C Program
{c_code}
## Output Format
Return the complete CUDA program inside a ```cuda code block.
Do not include any explanation outside the code block."""


def format_output(c_code: str) -> str:
    return USER_TRANSLATE_INITIAL.format(reference_section='', c_code=c_code)


USER_TRANSLATE_FIRST_PASS_CODE_ONLY = """Translate this C code to CUDA:

```c
{c_code}
```

Output ONLY the CUDA code in a ```cuda code block."""


USER_OPTIMIZE_WITH_PIPELINE_PROMPT = """{pipeline_opt_prompt}

Original C code (for reference):
```c
{c_code}
```

Output ONLY the fixed CUDA code in a ```cuda code block."""


USER_OPTIMIZE_FALLBACK = """Current CUDA code (needs fixing):
```cuda
{cuda_code}
```
{errors_section}
Original C code (for reference):
```c
{c_code}
```

Output ONLY the fixed CUDA code in a ```cuda code block."""


# ---------- Helpers ----------

def build_reference_section(kernels: List[str], paired_examples: Optional[List[dict]] = None) -> str:
    sections = []
    if paired_examples:
        sections.append('## Reference Examples (C--CUDA Pairs)')
        sections.extend(
            f"### Example {i}\n[Example C Code]\n{pair['c_code']}\n"
            f"[Example CUDA Kernel]\n{pair['cuda_code']}"
            for i, pair in enumerate(paired_examples, 1)
        )
    if kernels:
        sections.append('## Reference CUDA Kernels')
        sections.extend(
            f'### Reference {i}\n[Reference CUDA Kernel]\n{code}'
            for i, code in enumerate(kernels, 1)
        )
    return '\n'.join(sections) + ('\n' if sections else '')


def build_errors_section(errors: List[str]) -> str:
    if not errors:
        return ""
    body = "".join(f"- {e}\n" for e in errors)
    return f"\nErrors to fix:\n{body}"


def build_initial_translation_prompt(
    c_code: str,
    task_type: str,
    reference_kernels: List[str],
    paired_examples: Optional[List[dict]] = None,
) -> Tuple[str, str]:
    reference_section = build_reference_section(reference_kernels, paired_examples)
    user_prompt = USER_TRANSLATE_INITIAL.format(reference_section=reference_section, c_code=c_code)
    return SYSTEM_TRANSLATE_BASE, user_prompt


def build_optimization_prompt(
    cuda_code: str,
    c_code: str,
    pipeline_opt_prompt: str = "",
    errors: Optional[List[str]] = None,
    is_first_pass: bool = False,
) -> Tuple[str, str]:
    """根据是否有评估结果，组装 system/user prompt。

    - is_first_pass=True：首次翻译，使用 code-only 风格的简短模板。
    - is_first_pass=False：迭代优化，若 pipeline 给了 optimization_prompt 就走它，
      否则回退到通用 fix 模板。
    """
    if is_first_pass:
        return (
            SYSTEM_CODE_ONLY_GENERATOR,
            USER_TRANSLATE_FIRST_PASS_CODE_ONLY.format(c_code=c_code),
        )

    if pipeline_opt_prompt:
        user_prompt = USER_OPTIMIZE_WITH_PIPELINE_PROMPT.format(
            pipeline_opt_prompt=pipeline_opt_prompt,
            c_code=c_code,
        )
    else:
        user_prompt = USER_OPTIMIZE_FALLBACK.format(
            cuda_code=cuda_code,
            errors_section=build_errors_section(errors or []),
            c_code=c_code,
        )

    return SYSTEM_CODE_ONLY_OPTIMIZER, user_prompt
