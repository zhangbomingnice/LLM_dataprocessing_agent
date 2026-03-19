"""
Prompt 对齐器 — 将多版本模型输出按 prompt_id 做 inner join 对齐。
"""

from __future__ import annotations

import logging
from typing import Any

from cn_eval.data_loader.schema import Prompt, ModelOutput, AlignedPair

logger = logging.getLogger(__name__)


class PromptAligner:
    """按 prompt_id 将测试题集与多版本模型输出对齐。"""

    def __init__(self, prompts: list[Prompt]):
        self._prompt_map: dict[str, Prompt] = {p.prompt_id: p for p in prompts}

    def align_pair(
        self,
        baseline_outputs: list[ModelOutput],
        candidate_outputs: list[ModelOutput],
        baseline_version: str = "",
        candidate_version: str = "",
    ) -> list[AlignedPair]:
        """
        将 baseline 和 candidate 的输出按 prompt_id 对齐。
        只保留两者都存在的 prompt（inner join）。
        """
        base_map = {o.prompt_id: o for o in baseline_outputs}
        cand_map = {o.prompt_id: o for o in candidate_outputs}

        common_ids = set(base_map.keys()) & set(cand_map.keys())
        missing_base = set(cand_map.keys()) - set(base_map.keys())
        missing_cand = set(base_map.keys()) - set(cand_map.keys())

        if missing_base:
            logger.warning(
                "[Aligner] %d 条 prompt 缺少 baseline 输出，将跳过",
                len(missing_base),
            )
        if missing_cand:
            logger.warning(
                "[Aligner] %d 条 prompt 缺少 candidate 输出，将跳过",
                len(missing_cand),
            )

        pairs: list[AlignedPair] = []
        for pid in sorted(common_ids):
            prompt = self._prompt_map.get(pid)
            base_out = base_map[pid]
            cand_out = cand_map[pid]

            pairs.append(AlignedPair(
                prompt_id=pid,
                prompt_text=prompt.text if prompt else "",
                category=prompt.category if prompt else "",
                baseline_version=baseline_version or base_out.model_version,
                baseline_response=base_out.response,
                candidate_version=candidate_version or cand_out.model_version,
                candidate_response=cand_out.response,
                metadata={
                    "baseline_meta": base_out.metadata,
                    "candidate_meta": cand_out.metadata,
                },
            ))

        logger.info(
            "[Aligner] 对齐完成: %d 对 (baseline=%s, candidate=%s, 跳过=%d)",
            len(pairs),
            baseline_version or "auto",
            candidate_version or "auto",
            len(missing_base) + len(missing_cand),
        )
        return pairs

    def align_multi_version(
        self,
        version_outputs: dict[str, list[ModelOutput]],
    ) -> dict[str, dict[str, ModelOutput]]:
        """
        将多版本输出对齐为 {prompt_id: {version: ModelOutput}} 结构。
        仅保留所有版本都覆盖的 prompt。
        """
        version_maps = {
            ver: {o.prompt_id: o for o in outputs}
            for ver, outputs in version_outputs.items()
        }

        all_id_sets = [set(m.keys()) for m in version_maps.values()]
        if not all_id_sets:
            return {}
        common_ids = all_id_sets[0]
        for s in all_id_sets[1:]:
            common_ids &= s

        result: dict[str, dict[str, ModelOutput]] = {}
        for pid in sorted(common_ids):
            result[pid] = {ver: m[pid] for ver, m in version_maps.items()}

        total_unique = set()
        for m in version_maps.values():
            total_unique |= set(m.keys())

        logger.info(
            "[Aligner] 多版本对齐: %d 个共同 prompt / %d 个总 prompt (%d 个版本)",
            len(common_ids), len(total_unique), len(version_outputs),
        )
        return result

    def get_coverage_report(
        self,
        version_outputs: dict[str, list[ModelOutput]],
    ) -> dict[str, dict[str, Any]]:
        """生成各版本的覆盖率报告。"""
        all_prompt_ids = set(self._prompt_map.keys())
        report = {}
        for ver, outputs in version_outputs.items():
            out_ids = {o.prompt_id for o in outputs}
            covered = all_prompt_ids & out_ids
            report[ver] = {
                "total_prompts": len(all_prompt_ids),
                "covered": len(covered),
                "missing": len(all_prompt_ids - out_ids),
                "extra": len(out_ids - all_prompt_ids),
                "coverage": f"{len(covered) / max(len(all_prompt_ids), 1) * 100:.1f}%",
            }
        return report
