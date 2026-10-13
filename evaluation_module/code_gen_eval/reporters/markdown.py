"""
报告生成器 - 生成JSON和Markdown格式的评估报告
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import logging

from ..core.models import EvaluationReport

logger = logging.getLogger(__name__)


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, output_dir: str = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_json(self, report: EvaluationReport, filename: Optional[str] = None) -> Path:
        """保存为JSON格式报告"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report.model_name}_{timestamp}.json"
            
        output_path = self.output_dir / filename
        
        # 使用model_dump序列化Pydantic对象
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                json.loads(report.model_dump_json()),
                f,
                indent=2,
                ensure_ascii=False,
            )
            
        logger.info(f"JSON报告已保存: {output_path}")
        return output_path
        
    def generate_markdown(self, report: EvaluationReport, filename: Optional[str] = None) -> Path:
        """生成Markdown格式报告"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report.model_name}_{timestamp}.md"
            
        output_path = self.output_dir / filename
        
        md_content = self._build_markdown(report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
            
        logger.info(f"Markdown报告已保存: {output_path}")
        return output_path
        
    def _build_markdown(self, report: EvaluationReport) -> str:
        """构建Markdown内容"""
        s = SummaryStats(report)
        
        lines = []
        
        # 标题
        lines.append(f"# C2CUDA评估报告 - {report.model_name}")
        lines.append("")
        lines.append(f"**评估时间**: {report.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**总任务数**: {report.summary.total_tasks}")
        lines.append(f"**总耗时**: {report.total_duration_ms / 1000:.1f}s")
        lines.append("")
        
        # 概览摘要
        lines.append("## 📊 概览摘要")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| 编译成功 | {s.comp_success_count} / {s.total} ({s.comp_rate:.1f}%) |")
        lines.append(f"| 运行成功 | {s.run_success_count} / {s.comp_success_count} ({s.run_rate:.1f}%) |")
        lines.append(f"| 平均编译时间 | {s.mean_build_time:.1f}ms |")
        lines.append(f"| 平均运行时间 | {s.mean_runtime:.1f}ms |")
        lines.append("")
        
        # 编译失败分析
        if s.comp_failures:
            lines.append("## ❌ 编译失败分析")
            lines.append("")
            lines.append("| 错误类型 | 数量 |")
            lines.append("|----------|------|")
            for error_type, count in s.comp_failures.items():
                lines.append(f"| {error_type} | {count} |")
            lines.append("")
            
        # 运行失败分析
        if s.run_failures:
            lines.append("## ❌ 运行失败分析")
            lines.append("")
            lines.append("| 错误类型 | 数量 |")
            lines.append("|----------|------|")
            for error_type, count in s.run_failures.items():
                lines.append(f"| {error_type} | {count} |")
            lines.append("")
            
        # 按类型统计
        lines.append("## 📈 按任务类型统计")
        lines.append("")
        lines.append("| 类型 | 总数 | 编译成功 | 运行成功 |")
        lines.append("|------|------|----------|----------|")
        
        for task_type, stats in s.by_type.items():
            lines.append(
                f"| {task_type} | {stats['total']} | "
                f"{stats['comp_success']} ({stats['comp_rate']:.1f}%) | "
                f"{stats['run_success']} ({stats['run_rate']:.1f}%) |"
            )
        lines.append("")
            
        # 代码质量统计
        lines.append("## 📝 代码质量统计")
        lines.append("")
        lines.append("| 指标 | 平均值 |")
        lines.append("|------|--------|")
        lines.append(f"| 代码行数 | {s.mean_code_lines:.1f} |")
        lines.append(f"| Kernel数量 | {s.mean_kernel_count:.1f} |")
        lines.append(f"| 使用Shared Memory比例 | {s.shared_mem_rate:.1f}% |")
        lines.append(f"| 有边界检查比例 | {s.boundary_check_rate:.1f}% |")
        lines.append("")
        
        # 失败任务列表
        failed_tasks = [r for r in report.results if not r.success or r.compilation.status != "success"]
        if failed_tasks:
            lines.append("## 💥 失败任务列表")
            lines.append("")
            for task in failed_tasks[:50]:  # 最多显示50个
                error = task.evaluation_errors[0] if task.evaluation_errors else task.compilation.error_message
                lines.append(f"- `{task.task_type}/{task.task_id}`: {error}")
                
            if len(failed_tasks) > 50:
                lines.append(f"- ... 还有 {len(failed_tasks) - 50} 个失败任务")
            lines.append("")
            
        # 配置信息
        lines.append("## ⚙️ 评估配置")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(report.config, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")
        
        return "\n".join(lines)


class SummaryStats:
    """辅助计算汇总统计"""
    
    def __init__(self, report: EvaluationReport):
        self.report = report
        s = report.summary
        self.total = s.total_tasks
        self.comp_success_count = s.compilation_success_count
        self.comp_rate = s.compilation_success_rate
        self.run_success_count = s.run_success_count
        self.run_rate = s.run_success_rate
        self.comp_failures = s.compilation_failures
        self.run_failures = s.run_failures
        
        # 计算更详细的统计
        success_results = [r for r in report.results if r.compilation.status == "success"]
        
        self.mean_build_time = (
            sum(r.compilation.build_time_ms for r in success_results) / len(success_results)
            if success_results else 0
        )
        
        run_success_results = [r for r in success_results if r.run.status == "success" and r.run.runtime_mean_ms]
        self.mean_runtime = (
            sum(r.run.runtime_mean_ms for r in run_success_results) / len(run_success_results)
            if run_success_results else 0
        )
        
        # 代码质量统计
        with_metrics = [r for r in report.results if r.code_metrics]
        self.mean_code_lines = (
            sum(r.code_metrics.code_lines for r in with_metrics) / len(with_metrics)
            if with_metrics else 0
        )
        self.mean_kernel_count = (
            sum(r.code_metrics.kernel_count for r in with_metrics) / len(with_metrics)
            if with_metrics else 0
        )
        
        with_shared = [r for r in with_metrics if r.code_metrics.shared_memory_count > 0]
        self.shared_mem_rate = len(with_shared) / len(with_metrics) * 100 if with_metrics else 0
        
        with_boundary = [r for r in with_metrics if r.code_metrics.has_boundary_check]
        self.boundary_check_rate = len(with_boundary) / len(with_metrics) * 100 if with_metrics else 0
        
        # 按类型统计
        self.by_type = {}
        type_map = {}
        for r in report.results:
            t = r.task_type
            if t not in type_map:
                type_map[t] = []
            type_map[t].append(r)
            
        for t, results in type_map.items():
            total = len(results)
            comp_success = sum(1 for r in results if r.compilation.status == "success")
            run_success = sum(1 for r in results if r.run.status == "success")
            self.by_type[t] = {
                'total': total,
                'comp_success': comp_success,
                'comp_rate': comp_success / total * 100 if total > 0 else 0,
                'run_success': run_success,
                'run_rate': run_success / comp_success * 100 if comp_success > 0 else 0,
            }
