"""Stability AI Stable Image Inpaint 接口。"""

from __future__ import annotations

from io import BytesIO

import requests
from PIL import Image

from network_utils import configured_proxy_url, create_http_session, use_system_proxy
from prompts import NEGATIVE_PROMPT

STABILITY_INPAINT_URL = "https://api.stability.ai/v2beta/stable-image/edit/inpaint"
MAX_INPUT_PIXELS = 9_000_000


def _png_bytes(image: Image.Image, mode: str) -> bytes:
    buffer = BytesIO()
    image.convert(mode).save(buffer, format="PNG")
    return buffer.getvalue()


def _prepare_inputs(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    source = image.convert("RGB")
    source_mask = mask.convert("L").resize(source.size, Image.Resampling.NEAREST)
    if min(source.size) < 64:
        raise ValueError("图片每条边至少需要64像素")
    pixels = source.width * source.height
    if pixels > MAX_INPUT_PIXELS:
        scale = (MAX_INPUT_PIXELS / pixels) ** 0.5
        new_size = (max(64, int(source.width * scale)), max(64, int(source.height * scale)))
        source = source.resize(new_size, Image.Resampling.LANCZOS)
        source_mask = source_mask.resize(new_size, Image.Resampling.NEAREST)
    return source, source_mask


def stability_inpaint(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    api_key: str,
    seed: int = 0,
) -> Image.Image:
    """调用官方 Stable Image Inpaint，返回生成的整张候选图。"""
    if not api_key or not api_key.strip():
        raise ValueError("请填写 Stability AI API Key")
    source, source_mask = _prepare_inputs(image, mask)
    data = {
        "prompt": prompt.strip(),
        "negative_prompt": NEGATIVE_PROMPT,
        "grow_mask": "1",
        "output_format": "png",
        "seed": str(max(0, int(seed))),
    }
    response: requests.Response | None = None
    network_errors: list[str] = []
    proxy_configured = configured_proxy_url() is not None
    attempts = [True] if use_system_proxy() and proxy_configured else [False]
    if attempts == [False] and proxy_configured:
        attempts.append(True)
    for prefer_proxy in attempts:
        try:
            with create_http_session(prefer_proxy=prefer_proxy) as session:
                response = session.post(
                    STABILITY_INPAINT_URL,
                    headers={"Authorization": f"Bearer {api_key.strip()}", "Accept": "image/*"},
                    files={
                        "image": ("mural.png", _png_bytes(source, "RGB"), "image/png"),
                        "mask": ("mask.png", _png_bytes(source_mask, "L"), "image/png"),
                    },
                    data=data,
                    timeout=(20, 240),
                )
            break
        except requests.RequestException as exc:
            route = "系统代理" if prefer_proxy else "直连"
            network_errors.append(f"{route}：{exc}")
    if response is None:
        route_hint = "已自动尝试直连和Windows本机代理。" if proxy_configured else "未检测到可用Windows本机代理。"
        raise RuntimeError(f"Stability网络请求失败：{'；'.join(network_errors)}\n{route_hint}") from None
    if response.status_code != 200:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text[:1500]
        raise RuntimeError(f"Stability API错误（HTTP {response.status_code}）：{detail}")
    content_type = response.headers.get("Content-Type", "")
    if "image" not in content_type.lower():
        raise RuntimeError("Stability API没有返回图片，请检查账户、额度和请求参数")
    try:
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"无法读取Stability返回的图片：{exc}") from exc
