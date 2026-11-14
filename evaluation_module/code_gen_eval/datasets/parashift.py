"""
数据集加载器 - 加载基准测试数据集
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

import logging

logger = logging.getLogger(__name__)


class ParaShiftDataset:
    """配对语料数据集加载器"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")
            
        self._load_metadata()
        
    def _load_metadata(self):
        """加载所有任务的元数据"""
        self.tasks: Dict[str, Dict] = {}
        
        for type_dir in sorted(self.dataset_path.iterdir()):
            if not type_dir.is_dir() or not type_dir.name.startswith("type-"):
                continue
                
            type_name = type_dir.name
            
            for task_dir in sorted(type_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                    
                task_id = task_dir.name
                meta_path = task_dir / "metadata.json"
                
                if meta_path.exists():
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                else:
                    metadata = {}
                    
                # 查找C和CUDA文件
                c_dir = task_dir / "c"
                cuda_dir = task_dir / "cuda"
                
                # 查找 .c 和 .cpp 文件
                c_files = []
                if c_dir.exists():
                    c_files.extend(list(c_dir.glob("*.c")))
                    c_files.extend(list(c_dir.glob("*.cpp")))
                
                cuda_files = list(cuda_dir.glob("*.cu")) if cuda_dir.exists() else []
                
                self.tasks[f"{type_name}/{task_id}"] = {
                    'task_id': task_id,
                    'task_type': type_name,
                    'metadata': metadata,
                    'c_path': str(c_files[0]) if c_files else None,
                    'cuda_path': str(cuda_files[0]) if cuda_files else None,
                }
                
        logger.info(f"加载数据集完成: {len(self.tasks)} 个任务")
        for t in ['type-1', 'type-2', 'type-3', 'type-4', 'type-5']:
            count = sum(1 for k in self.tasks if k.startswith(t))
            if count > 0:
                logger.info(f"  {t}: {count} 个任务")
                
    def get_tasks_by_type(self, task_type: str) -> List[Dict]:
        """按类型获取任务"""
        return [
            task for key, task in self.tasks.items()
            if task['task_type'] == task_type
        ]
        
    def get_task(self, task_key: str) -> Optional[Dict]:
        """获取单个任务"""
        return self.tasks.get(task_key)
        
    def list_all_tasks(self) -> List[str]:
        """列出所有任务key"""
        return sorted(self.tasks.keys())
