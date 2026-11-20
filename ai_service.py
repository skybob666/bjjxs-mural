"""DeepSeek 图像分析服务和本地演示分析。"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from io import BytesIO
from typing import Any

from PIL import Image, ImageOps
import requests

from image_utils import validate_bbox
from network_utils import configured_proxy_url, create_http_session, use_system_proxy
from prompts import SYSTEM_PROMPT


TRANSIENT_HTTP_CODES = {500, 502, 503, 504, 550}
API_BASE_URL = "https://api.deepseek.com"
# 图像观察是交互操作；失败时应尽快返回明确原因，而不是长时间卡住页面。
API_TIMEOUT = (8, 25)
CONNECTION_TEST_TIMEOUT = (5, 12)


def _response_detail(response: requests.Response) -> str:
    try:
        detail: Any = response.json()
    except ValueError:
        detail = response.text.strip()
    if not detail:
        detail = "服务端未返回错误正文"
    trace_id = (
        response.headers.get("x-request-id")
        or response.headers.get("x-ds-trace-id")
        or response.headers.get("request-id")
    )
    if trace_id:
        return f"{detail}；请求标识：{trace_id}"
    return str(detail)


def _request_deepseek(
    method: str,
    path: str,
    api_key: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: tuple[int, int] = API_TIMEOUT,
) -> tuple[requests.Response, str]:
    """用可预期的路由顺序请求 DeepSeek，并保留明确的失败上下文。"""
    proxy_enabled = use_system_proxy() and configured_proxy_url() is not None
    # 默认只直连，避免不可用的 Windows 代理让一次点击额外等待几十秒。
    # 用户显式开启代理后，才先走代理；代理失败仍会回退到直连。
    attempts = [True, False] if proxy_enabled else [False]
    network_errors: list[str] = []
    last_response: requests.Response | None = None
    last_route = "直连"

    for prefer_proxy in attempts:
        route = "系统代理" if prefer_proxy else "直连"
        for retry_index in range(2):
            try:
                with create_http_session(prefer_proxy=prefer_proxy) as session:
                    request = getattr(session, method.lower())
                    request_kwargs: dict[str, Any] = {
                        "headers": {
                            "Authorization": f"Bearer {api_key.strip()}",
                            "Content-Type": "application/json",
                        },
                        "timeout": timeout,
                    }
                    if payload is not None:
                        request_kwargs["json"] = payload
                    response = request(f"{API_BASE_URL}{path}", **request_kwargs)
            except requests.RequestException as exc:
                network_errors.append(f"{route}：{exc}")
                break

            last_response, last_route = response, route
            if response.status_code not in TRANSIENT_HTTP_CODES:
                return response, route
            if retry_index == 0:
                time.sleep(0.8)

    if last_response is not None:
        return last_response, last_route

    route_hint = "当前仅使用直连；如校园网络必须经过已验证代理，可在 .env 中设置 WALL_MURAL_USE_SYSTEM_PROXY=1。"
    if proxy_enabled:
        route_hint = "已依次尝试系统代理和直连。"
    details = "；".join(network_errors) or "未获得有效响应"
    raise RuntimeError(f"DeepSeek网络请求失败：{details}。{route_hint}") from None


def test_deepseek_connection(api_key: str) -> dict[str, Any]:
    """低成本检查 Key、网络和当前可用模型，不发送图片或消耗推理额度。"""
    if not api_key or not api_key.strip():
        raise ValueError("请先填写 DeepSeek API Key")

    started = time.perf_counter()
    response, route = _request_deepseek(
        "GET",
        "/models",
        api_key,
        timeout=CONNECTION_TEST_TIMEOUT,
    )
    elapsed = round(time.perf_counter() - started, 2)
    if response.status_code != 200:
        raise RuntimeError(f"DeepSeek连接测试失败（HTTP {response.status_code}）：{_response_detail(response)}")
    try:
        models = [item.get("id") for item in response.json().get("data", []) if item.get("id")]
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("DeepSeek连接成功，但模型列表格式无法识别。") from exc
    return {"models": models, "route": route, "elapsed_seconds": elapsed}


def image_to_data_url(image: Image.Image, max_side: int = 1024, quality: int = 82) -> str:
    """将本地图像压缩为适合 API 传输的 JPEG data URL。"""
    prepared = ImageOps.exif_transpose(image).convert("RGB").copy()
    prepared.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    prepared.save(buffer, format="JPEG", quality=quality, optimize=True, progressive=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _extract_json(text: str) -> dict[str, Any]:
    """兼容模型偶尔附带代码围栏或解释文字的情况。"""
    candidate = (text or "").strip()
    candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
    candidate = re.sub(r"\s*```$", "", candidate)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("AI返回内容中没有找到JSON对象") from None
        try:
            value = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"AI返回结果无法解析为JSON：{exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("AI返回的数据不是JSON对象")
    return value


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def build_recommendations(analysis: dict[str, Any]) -> list[dict[str, str]]:
    """
    把建议组织成规则第2条要求的“建议—依据—风险”三件套。

    优先使用已有的 recommendations；否则从 AI 输出的 protection_suggestions、
    general_risks 以及各区域的 evidence / uncertainty 现场合成。这样无论是实时
    DeepSeek、本地演示还是兜底数据，观察卡都能稳定显示三件套。
    """
    raw = analysis.get("recommendations")
    if isinstance(raw, list):
        clean: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            action = _text(item.get("action") or item.get("suggestion")).strip()
            if not action:
                continue
            clean.append(
                {
                    "category": _text(item.get("category"), "保护记录").strip() or "保护记录",
                    "action": action,
                    "basis": _text(item.get("basis")).strip(),
                    "risk": _text(item.get("risk")).strip(),
                }
            )
        if clean:
            return clean[:2]

    suggestions = analysis.get("protection_suggestions", [])
    if not isinstance(suggestions, list):
        suggestions = [suggestions] if suggestions else []
    suggestions = [s for s in (_text(i).strip() for i in suggestions) if s][:2]

    risks = analysis.get("general_risks", [])
    if not isinstance(risks, list):
        risks = [risks] if risks else []
    risks = [r for r in (_text(i).strip() for i in risks) if r]

    regions = [
        r
        for r in analysis.get("regions", [])
        if isinstance(r, dict) and r.get("status") != "已驳回"
    ]
    evidences: list[str] = []
    uncertainties: list[str] = []
    for region in regions:
        for key, bucket in (("evidence", evidences), ("uncertainty", uncertainties)):
            value = " ".join(_text(region.get(key)).split())
            if value and value not in bucket:
                bucket.append(value)

    basis_all = "；".join(evidences[:2])
    basis_review = "；".join(uncertainties[:2])
    risk_image = risks[0] if risks else ""
    risk_expert = next((r for r in risks if any(k in r for k in ("专家", "专业", "鉴定"))), "")
    if not risk_expert:
        risk_expert = risks[2] if len(risks) > 2 else (risks[-1] if risks else "")

    fallback_basis = "观察卡记录的可见现象与人工复核状态。"
    fallback_risk = "单张图像受光照、分辨率影响，存在误判或漏判可能。"

    result: list[dict[str, str]] = []
    for index, action in enumerate(suggestions):
        category = "数字展示" if any(k in action for k in ("数字", "展示", "传播")) else "保护记录"
        if index == 0:
            basis = basis_all or fallback_basis
            risk = risk_image or fallback_risk
        else:
            basis = basis_review or basis_all or fallback_basis
            risk = risk_expert or risk_image or fallback_risk
        result.append({"category": category, "action": action, "basis": basis, "risk": risk})
    return result


def normalize_result(data: dict[str, Any]) -> dict[str, Any]:
    """清洗模型输出，保证前端拿到稳定的数据结构。"""
    clean_regions: list[dict[str, Any]] = []
    regions = data.get("regions", [])
    if not isinstance(regions, list):
        regions = []

    for index, region in enumerate(regions[:3]):
        if not isinstance(region, dict):
            continue
        raw_bbox = region.get("bbox")
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            continue
        try:
            bbox = [float(item) for item in raw_bbox]
        except (TypeError, ValueError):
            continue
        bbox = validate_bbox(bbox)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        clean_regions.append(
            {
                "id": index + 1,
                "bbox": [x1, y1, x2, y2],
                "ai_bbox": [x1, y1, x2, y2],
                "visible_observation": _text(region.get("visible_observation")),
                "suspected_issue": _text(region.get("suspected_issue"), "未知异常区域"),
                "evidence": _text(region.get("evidence")),
                "uncertainty": _text(region.get("uncertainty"), "需要人工和专业人员复核"),
                "recommendation": _text(region.get("recommendation")),
                "source": "AI候选",
                "status": "待复核",
                "human_note": "",
                "human_review_required": True,
                "human_reviewed": bool(region.get("human_reviewed", False)),
            }
        )

    suggestions = data.get("protection_suggestions", [])
    risks = data.get("general_risks", [])
    if not isinstance(suggestions, list):
        suggestions = [str(suggestions)] if suggestions else []
    if not isinstance(risks, list):
        risks = [str(risks)] if risks else []

    result = {
        "overall_observation": _text(data.get("overall_observation"), "未提供整体观察"),
        "regions": clean_regions,
        "protection_suggestions": [_text(item) for item in suggestions[:2]],
        "general_risks": [_text(item) for item in risks[:8]],
    }
    result["recommendations"] = build_recommendations(
        {**result, "recommendations": data.get("recommendations")}
    )
    return result


def analyze_mural(
    image: Image.Image,
    api_key: str,
    model: str | None = None,
) -> dict[str, Any]:
    """调用 DeepSeek Vision，返回经过清洗的结构化分析结果。"""
    if not api_key or not api_key.strip():
        raise ValueError("请填写 DeepSeek API Key")

    selected_model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    image_url = image_to_data_url(image)
    image_detail = os.getenv("DEEPSEEK_IMAGE_DETAIL", "low").strip().lower()
    if image_detail not in {"low", "high", "original", "auto"}:
        image_detail = "low"
    payload = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "请分析这张壁画图片。最多提出3处值得人工复核的区域，"
                            "严格按要求输出JSON，并使用0到1000的归一化坐标；不要输出Markdown。"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": image_detail},
                    },
                ],
            },
        ],
        "temperature": 0.2,
        # 观察卡需要短、稳定的 JSON；关闭推理模式可显著减少等待时间。
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
        "max_tokens": 1200,
    }
    response, _route = _request_deepseek("POST", "/chat/completions", api_key, payload=payload)
    if response.status_code != 200:
        detail = _response_detail(response)
        if response.status_code in TRANSIENT_HTTP_CODES:
            detail = f"服务端或上游网关暂时异常，已完成有限重试；{detail}"
        raise RuntimeError(f"DeepSeek API错误（HTTP {response.status_code}）：{detail}")
    try:
        response_data = response.json()
        choice = response_data["choices"][0]
        message = choice["message"].get("content") or ""
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(f"DeepSeek返回结构异常：{response.text[:1500]}") from exc
    if not message.strip():
        raise ValueError("AI返回内容为空，请重试")
    if choice.get("finish_reason") == "length":
        raise ValueError("AI输出被截断，请重试或减少图片尺寸")
    return normalize_result(_extract_json(message))


def demo_analysis(image: Image.Image, synthetic: bool = False) -> dict[str, Any]:
    """无外部API时的流程数据；真实图片使用保守的人工待观察框。"""
    width, height = image.size
    regions = [
        {
            "id": 1,
            "bbox": [475, 525, 570, 675],
            "ai_bbox": [475, 525, 570, 675],
            "visible_observation": "局部色块连续性较弱，边缘存在明暗变化。",
            "suspected_issue": "疑似颜料层缺失或表面磨损",
            "evidence": "仅依据当前图像中的色差和纹理变化进行初步观察。",
            "uncertainty": "需要原图高清细节、现场光照记录和专业人员确认。",
            "recommendation": "先记录区域位置和图像条件，避免将色差直接解释为病害成因。",
            "source": "本地演示",
            "status": "待复核",
            "human_note": "",
            "human_review_required": True,
            "human_reviewed": False,
        },
        {
            "id": 2,
            "bbox": [700, 350, 850, 510],
            "ai_bbox": [700, 350, 850, 510],
            "visible_observation": "局部纹理呈不均匀块状分布，可能受到图像压缩或光照影响。",
            "suspected_issue": "疑似表面污染或图像采集伪影",
            "evidence": "可见区域与邻近画面在颜色和纹理上存在差异。",
            "uncertainty": "单张图片无法排除拍摄反光、阴影和压缩噪声。",
            "recommendation": "建议补充不同光照条件下的照片，并由专业人员复核。",
            "source": "本地演示",
            "status": "待复核",
            "human_note": "",
            "human_review_required": True,
            "human_reviewed": False,
        },
    ]
    if not synthetic:
        regions = [
            {
                "id": 1,
                "bbox": [350, 300, 650, 700],
                "ai_bbox": None,
                "visible_observation": "当前为本地兜底流程，未对图片作真实病害判断。",
                "suspected_issue": "待人工观察区域",
                "evidence": "该区域仅用于保证人工复核和导出流程可继续，不是模型识别结果。",
                "uncertainty": "没有可用的实时AI结果；需要人工结合原图重新框选和记录。",
                "recommendation": "请人工拖拽框选真正需要观察的区域，并补充图像来源和现场信息。",
                "source": "本地兜底",
                "status": "待复核",
                "human_note": "",
                "human_review_required": True,
                "human_reviewed": False,
            }
        ]
    elif width < 640 or height < 480:
        regions = regions[:1]
    demo_result = {
        "overall_observation": (
            f"本地演示已读取一张 {width}×{height} 像素图像。"
            "以下区域仅用于演示‘AI提出候选—人工复核’的流程，"
            "不代表真实病害识别结果。"
        ),
        "regions": regions,
        "protection_suggestions": [
            "保留原始图像和拍摄条件，建立可追溯的图像记录。",
            "任何病害成因判断和实体修复方案都应交由相关专业人员确认。",
        ],
        "general_risks": [
            "本地演示未调用外部模型，结果不是模型推理结果。",
            "图像中的色差也可能来自光照、压缩、反光或拍摄设备。",
        ],
    }
    demo_result["recommendations"] = build_recommendations(demo_result)
    return demo_result


def fallback_analysis(image: Image.Image, reason: str) -> dict[str, Any]:
    """API失败时的明确降级结果，绝不冒充实时AI输出。"""
    result = demo_analysis(image, synthetic=False)
    short_reason = " ".join(str(reason).split())[:500]
    result["analysis_mode"] = "本地兜底数据（DeepSeek未完成，未冒充AI）"
    result["fallback_reason"] = short_reason
    result["overall_observation"] = (
        f"本次图片已成功读取，尺寸为 {image.width}×{image.height} 像素；"
        "DeepSeek实时观察未完成，系统切换到本地兜底，仅用于继续人工框选、复核和导出。"
    )
    result["general_risks"] = [
        f"实时AI未完成：{short_reason}",
        "本地兜底区域不是模型推理结果，不得在答辩中表述为AI识别结论。",
        "建议检查网络、API余额、模型权限后再进行真实AI观察。",
    ]
    result["recommendations"] = build_recommendations(result)
    return result
