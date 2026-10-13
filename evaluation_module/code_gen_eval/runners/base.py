"""
基础运行器 - 封装编译和运行的通用逻辑
"""

import os
import subprocess
import time
import shlex
from pathlib import Path
from typing import Tuple, Optional, List

import logging

logger = logging.getLogger(__name__)


class BaseRunner:
    """运行器基类"""
    
    def __init__(self, workdir: Path, timeout: int = 120):
        self.workdir = workdir
        self.timeout = timeout
        self.workdir.mkdir(parents=True, exist_ok=True)
        
    def run_command(
        self,
        cmd: List[str],
        cwd: Optional[Path] = None,
        capture_output: bool = True,
        timeout: Optional[int] = None,
        env: Optional[dict] = None,
    ) -> Tuple[int, str, str, float]:
        """
        执行shell命令
        返回: (return_code, stdout, stderr, elapsed_seconds)
        
        Args:
            env: 额外的环境变量字典
        """
        cwd = cwd or self.workdir
        timeout = timeout or self.timeout
        
        cmd_str = " ".join(shlex.quote(c) for c in cmd)
        logger.debug(f"执行命令: {cmd_str}")
        
        start_time = time.perf_counter()
        
        # 合并环境变量
        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(cwd),
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                env=process_env,
            )
            elapsed = time.perf_counter() - start_time
            
            return (
                result.returncode,
                result.stdout or "",
                result.stderr or "",
                elapsed,
            )
            
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start_time
            logger.warning(f"命令超时 ({timeout}s): {cmd_str}")
            return -1, "", f"TIMEOUT: command exceeded {timeout}s", elapsed
            
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(f"命令执行失败: {e}")
            return -1, "", str(e), elapsed
