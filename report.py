"""报告、JSON、图片和观察卡成果导出。"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image
from review_evidence import card_recommendations, card_regions, current_review


def _safe_name(name: str) -> str:
    stem = Path(name).stem or "mural"
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", stem)[:60]


def build_report(
    image_name: str,
    image_source: str,
    analysis: dict[str, Any],
    analysis_mode: str = "未说明",
    task: dict[str, Any] | None = None,
    audit_log: list[dict[str, Any]] | None = None,
) -> str:
    task = task or {}
    regions = analysis.get("regions", [])
    active = card_regions(analysis)
    reviewed = sum(bool(current_review(region)) for region in active)
    lines = [
        "# 壁鉴新生——壁画智能观察报告",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"任务编号：{task.get('task_id', '待编号')}",
        f"任务标题：{task.get('title', '未命名壁画观察任务')}",
        f"原始图像：{image_name}",
        f"图像来源：{image_source or task.get('image_source') or '待补充'}",
        f"授权说明：{task.get('authorization', '待补充')}",
        f"分析模式：{analysis_mode}",
        f"过程记录：{len(audit_log or [])} 条；当前框已有证据：{reviewed}/{len(active)}",
        "",
        "> 重要说明：以下内容是图像观察辅助结果，不构成文物病害专业鉴定。",
        "> 本项目不做虚拟修复：对已丢失的壁画内容不做无依据补全，只做图像观察、疑似标注与人工复核。观察结论不构成文物病害专业鉴定。",
        "",
        "## 一、整体观察",
        "",
        str(analysis.get("overall_observation", "未提供")),
        "",
        "## 二、疑似异常区域",
        "",
    ]
    if not regions:
        lines.append("未提供有效区域；请结合原图进行人工观察。")
    for region in regions:
        review = current_review(region)
        coordinate_lines = []
        if region.get("ai_bbox"):
            coordinate_lines.append(f"- AI原始坐标：{region.get('ai_bbox')}")
        coordinate_lines.append(f"- 当前复核坐标：{region.get('bbox', [])}")
        lines.extend(
            [
                f"### 区域{region.get('id', '')}（{region.get('status', '待复核')}）",
                f"- 来源：{region.get('source', 'AI候选')}",
                *coordinate_lines,
                f"- 原候选描述：{region.get('visible_observation', '')}",
                f"- 原候选疑似问题：{region.get('suspected_issue', '')}",
                f"- 原候选依据：{region.get('evidence', '')}",
                f"- 当前框图像证据：{review.get('observation') or '待核对'}",
                *([f"- 驳回理由：{review['reason']}"] if region.get("status") == "已驳回" and review.get("reason") else []),
                *([f"- 当前框待核点：{review['uncertainty']}"] if review.get("uncertainty") else []),
                "",
            ]
        )
    lines += ["## 三、建议 / 依据 / 可能风险", ""]
    recommendations = card_recommendations(analysis)
    if recommendations:
        for index, rec in enumerate(recommendations, start=1):
            lines.extend(
                [
                    f"### 建议{index}（{rec['id']}）",
                    f"- 建议：{rec.get('action', '')}",
                    f"- 依据：{rec.get('basis', '') or '见整体观察与区域证据'}",
                    f"- 可能风险：{rec.get('risk', '') or '结论需专业人员确认'}",
                    "",
                ]
            )
    else:
        lines.append("待填写与保留区域对应的建议及风险。")
    lines += ["", "## 四、通用风险与限制", ""]
    for item in analysis.get("general_risks", []):
        lines.append(f"- {item}")
    lines += ["", "## 五、人工复核与边界", "", "- 所有疑似区域均需人工逐条确认、修改或驳回后方可写入观察卡。", "- AI 提供的是疑似观察候选，不是病害鉴定；缺失内容不做无依据补全。"]
    return "\n".join(lines)


def export_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def image_bytes(image: Image.Image, image_format: str = "PNG") -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def save_outputs(
    output_dir: str | Path,
    image_name: str,
    original: Image.Image,
    annotated: Image.Image,
    analysis_payload: dict[str, Any],
    report_text: str,
    pdf_bytes: bytes | None = None,
) -> Path:
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    prefix = f"{_safe_name(image_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    original.save(folder / f"{prefix}_original.png")
    annotated.save(folder / f"{prefix}_regions.png")
    (folder / f"{prefix}_analysis.json").write_text(export_json(analysis_payload), encoding="utf-8")
    (folder / f"{prefix}_report.md").write_text(report_text, encoding="utf-8")
    if pdf_bytes:
        (folder / f"{prefix}_observation_card.pdf").write_bytes(pdf_bytes)
    return folder
