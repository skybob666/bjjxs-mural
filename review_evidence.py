"""将人工观察、复核理由和下一步检查绑定到当前框。"""

from image_utils import validate_bbox


def region_code(region):
    return f"R{int(region['id']):02d}"


def review_at_bbox(region):
    saved = region.get("review_evidence")
    bbox = validate_bbox(region.get("bbox"))
    if isinstance(saved, dict) and bbox and saved.get("bbox") == bbox:
        return saved
    return {}


def current_review(region):
    saved = review_at_bbox(region)
    status = region.get("status")
    if status == "已确认" and saved.get("observation"):
        return saved
    if status == "已驳回" and saved.get("reason"):
        return saved
    return {}


def save_region_review(region, bbox, status, observation="", uncertainty="", action="", risk="", *, rejection_reason=""):
    bbox = validate_bbox(bbox)
    if bbox is None:
        raise ValueError("框坐标无效，请调整四个边界。")
    if status not in {"已确认", "已驳回"}:
        raise ValueError("请选择确认或驳回。")
    observation, uncertainty, action, risk, rejection_reason = (
        " ".join(value.split()) for value in (observation, uncertainty, action, risk, rejection_reason)
    )
    if status == "已驳回":
        if not rejection_reason:
            raise ValueError("请选择或填写驳回理由。")
        evidence = {"bbox": bbox.copy(), "reason": rejection_reason}
    else:
        if not observation:
            raise ValueError("请选择或填写框内可见现象。")
        if bool(action) != bool(risk):
            raise ValueError("填写建议时请同时填写可能风险，也可将两栏留空。")
        evidence = {
            "bbox": bbox.copy(), "observation": observation, "uncertainty": uncertainty,
            "action": action, "risk": risk,
        }
    region.update(
        bbox=bbox,
        status=status,
        human_note=evidence.get("reason", ""),
        human_reviewed=True,
        review_evidence=evidence,
    )


def card_regions(analysis):
    return [
        region for region in analysis.get("regions", [])
        if region.get("status") != "已驳回" and validate_bbox(region.get("bbox"))
    ]


def card_recommendations(analysis):
    """只引用当前仍保留、已按框记录的观察依据。"""
    result = []
    for region in card_regions(analysis):
        review = current_review(region)
        if review.get("action") and review.get("risk") and review.get("observation"):
            result.append({
                "id": region_code(region),
                "action": review["action"],
                "basis": review["observation"],
                "risk": review["risk"],
            })
    return result


def card_issues(analysis, task=None):
    regions = card_regions(analysis)
    issues = []
    if not 1 <= len(regions) <= 3:
        issues.append("线上观察卡需保留 1–3 处有效区域。")
    missing = [region_code(region) for region in regions if not current_review(region)]
    if missing:
        issues.append(f"当前框尚未保存对应证据：{'、'.join(missing)}。")
    if not 1 <= len(card_recommendations(analysis)) <= 2:
        issues.append("请在保留区域中填写 1–2 条带风险的建议。")
    if task and any(str(task.get(key) or "").strip() in {"", "待补充"} for key in ("image_source", "authorization")):
        issues.append("请补充图像来源和使用说明。")
    mode = str(analysis.get("analysis_mode") or "")
    if "兜底" in mode or "本地演示" in mode:
        issues.append("本次使用本地流程。正式视频需有真实 AI 操作及同一轮结果。")
    return issues
