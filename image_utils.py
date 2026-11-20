"""图片尺寸、候选区域、绘制遮罩和本地演示图工具。"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


MIN_BBOX_SIZE = 12


def resize_image(image: Image.Image, max_side: int = 1200) -> Image.Image:
    result = image.convert("RGB").copy()
    result.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return result


def validate_bbox(bbox: Any, min_size: int = MIN_BBOX_SIZE) -> list[int] | None:
    """验证严格的0-1000归一化坐标；不会把0-1误当成1000坐标。"""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        values = [int(round(float(item))) for item in bbox]
    except (TypeError, ValueError):
        return None
    if any(value < 0 or value > 1000 for value in values):
        return None
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1 or x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    return values


def clamp_bbox(bbox: Any) -> list[int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        values = [max(0, min(1000, int(round(float(item))))) for item in bbox]
    except (TypeError, ValueError):
        return None
    return validate_bbox(values)


def bbox_to_pixels(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    checked = clamp_bbox(bbox) or [0, 0, 0, 0]
    x1, y1, x2, y2 = checked
    return (
        int(x1 / 1000 * max(0, width - 1)),
        int(y1 / 1000 * max(0, height - 1)),
        int(x2 / 1000 * max(0, width - 1)),
        int(y2 / 1000 * max(0, height - 1)),
    )


def crop_region(image: Image.Image, bbox: list[int]) -> Image.Image:
    left, top, right, bottom = bbox_to_pixels(bbox, *image.size)
    return image.crop((left, top, right + 1, bottom + 1))


def pixels_to_bbox(rect: tuple[float, float, float, float], width: int, height: int) -> list[int] | None:
    """把画布像素矩形转换为0-1000坐标。"""
    if width <= 0 or height <= 0:
        return None
    left, top, right, bottom = rect
    x1, x2 = sorted((float(left), float(right)))
    y1, y2 = sorted((float(top), float(bottom)))
    return clamp_bbox([
        x1 / width * 1000,
        y1 / height * 1000,
        x2 / width * 1000,
        y2 / height * 1000,
    ])


def canvas_size_for_image(image: Image.Image, max_width: int = 900, max_height: int = 620) -> tuple[int, int]:
    ratio = min(max_width / image.width, max_height / image.height, 1.0)
    return max(320, int(image.width * ratio)), max(240, int(image.height * ratio))


def fabric_rectangles_to_bboxes(json_data: dict[str, Any] | None, canvas_size: tuple[int, int], image_size: tuple[int, int]) -> list[list[int]]:
    """读取 drawable-canvas 的矩形对象，映射回原图坐标。"""
    if isinstance(json_data, str):
        try:
            json_data = json.loads(json_data)
        except (TypeError, ValueError):
            return []
    if not isinstance(json_data, dict):
        return []
    canvas_width, canvas_height = canvas_size
    image_width, image_height = image_size
    scale_x = image_width / max(1, canvas_width)
    scale_y = image_height / max(1, canvas_height)
    result: list[list[int]] = []
    for item in json_data.get("objects", []):
        shape_type = str(item.get("type", "")).lower() if isinstance(item, dict) else ""
        if not isinstance(item, dict) or shape_type not in {"rect", "rectangle"}:
            continue
        left = float(item.get("left", 0))
        top = float(item.get("top", 0))
        width = float(item.get("width", 0)) * float(item.get("scaleX", 1))
        height = float(item.get("height", 0)) * float(item.get("scaleY", 1))
        # Fabric 在不同版本中可能使用中心点或右下角作为对象原点。
        if item.get("originX") == "center":
            left -= width / 2
        elif item.get("originX") == "right":
            left -= width
        if item.get("originY") == "center":
            top -= height / 2
        elif item.get("originY") == "bottom":
            top -= height
        bbox = pixels_to_bbox((left * scale_x, top * scale_y, (left + width) * scale_x, (top + height) * scale_y), image_width, image_height)
        if bbox:
            result.append(bbox)
    return result


def canvas_delta_to_bbox(
    canvas_image: Any,
    background: Image.Image,
    canvas_size: tuple[int, int],
    image_size: tuple[int, int],
    threshold: int = 18,
) -> list[int] | None:
    """当画布组件未返回 JSON 时，从带标记的画布图像反推一个候选矩形。

    这是 JSON 读取失败时的兜底，不会替代正常的 Fabric 对象读取。仅接受面积合理的
    前景差异，避免把整张背景图误判为人工框选。
    """
    if canvas_image is None:
        return None
    canvas_width, canvas_height = canvas_size
    image_width, image_height = image_size
    if min(canvas_width, canvas_height, image_width, image_height) <= 0:
        return None
    try:
        overlay = Image.fromarray(canvas_image).convert("RGB").resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
    except Exception:
        return None
    base = background.convert("RGB").resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
    delta = ImageChops.difference(overlay, base).convert("L")
    changed = delta.point(lambda value: 255 if value >= threshold else 0)
    bbox = changed.getbbox()
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    changed_area = max(0, right - left) * max(0, bottom - top)
    canvas_area = canvas_width * canvas_height
    # 画布整体变化通常代表背景重采样差异，不能误当作用户画框。
    if changed_area / max(1, canvas_area) > 0.92:
        return None
    scale_x = image_width / canvas_width
    scale_y = image_height / canvas_height
    return pixels_to_bbox((left * scale_x, top * scale_y, right * scale_x, bottom * scale_y), image_width, image_height)


def draw_regions(image: Image.Image, regions: list[dict], *, number_only: bool = False) -> Image.Image:
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.load_default(size=16) if number_only else ImageFont.load_default()
    except TypeError:
        font = ImageFont.load_default()
    width, height = result.size
    colors = [(161, 59, 43), (30, 119, 125), (198, 149, 60)]
    for region in regions:
        if region.get("status") == "已驳回":
            continue
        bbox = validate_bbox(region.get("bbox"))
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox_to_pixels(bbox, width, height)
        color = colors[(int(region.get("id", 1)) - 1) % len(colors)]
        if region.get("human_reviewed"):
            color = (198, 149, 60)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(3, width // 300))
        label = f"R{int(region.get('id', 1)):02d}"
        if not number_only:
            label += f" · {region.get('status', '待复核')}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        label_width = right - left + 12
        label_height = bottom - top + 8
        label_y = max(0, y1 - label_height)
        label_x = max(0, min(x1, width - label_width))
        draw.rounded_rectangle((label_x, label_y, label_x + label_width, label_y + label_height), radius=4, fill=color)
        draw.text((label_x + 6, label_y + 4 - top), label, fill=(255, 255, 255), font=font)
    return result


def create_mask(image_size: tuple[int, int], bbox: list[float], padding_px: int = 6) -> Image.Image:
    width, height = image_size
    mask = Image.new("L", (width, height), 0)
    checked = validate_bbox(bbox)
    if checked is None:
        return mask
    draw = ImageDraw.Draw(mask)
    x1, y1, x2, y2 = bbox_to_pixels(checked, width, height)
    pad = max(0, int(padding_px))
    draw.rectangle((max(0, x1 - pad), max(0, y1 - pad), min(width - 1, x2 + pad), min(height - 1, y2 + pad)), fill=255)
    return mask


def mask_from_canvas_delta(canvas_image: Any, background: Image.Image, target_size: tuple[int, int], threshold: int = 18) -> Image.Image:
    """从画布前后差异生成自由绘制遮罩。"""
    if canvas_image is None:
        return Image.new("L", target_size, 0)
    overlay = Image.fromarray(canvas_image).convert("RGBA").resize(background.size, Image.Resampling.LANCZOS)
    base = background.convert("RGBA")
    delta = ImageChops.difference(overlay, base).convert("L")
    delta = delta.point(lambda value: 255 if value >= threshold else 0)
    delta = delta.filter(ImageFilter.MaxFilter(5))
    return delta.resize(target_size, Image.Resampling.NEAREST)


def mask_area_ratio(mask: Image.Image) -> float:
    gray = mask.convert("L")
    total = gray.width * gray.height
    return sum(1 for value in gray.getdata() if value > 20) / max(1, total)


def overlay_mask(image: Image.Image, mask: Image.Image, color: tuple[int, int, int] = (161, 59, 43)) -> Image.Image:
    base = image.convert("RGBA")
    tint = Image.new("RGBA", base.size, (*color, 88))
    alpha = mask.convert("L").resize(base.size, Image.Resampling.NEAREST)
    tint.putalpha(alpha.point(lambda value: int(value * 0.38)))
    return Image.alpha_composite(base, tint).convert("RGB")


def compose_inpaint(original: Image.Image, generated: Image.Image, mask: Image.Image) -> Image.Image:
    """仅在遮罩内部合成生成结果，避免API返回图改动未选区域。"""
    base = original.convert("RGB")
    candidate = generated.convert("RGB").resize(base.size, Image.Resampling.LANCZOS)
    mask = mask.convert("L").resize(base.size, Image.Resampling.NEAREST)
    return Image.composite(candidate, base, mask)


def outside_mask_change_score(original: Image.Image, candidate: Image.Image, mask: Image.Image) -> float:
    base = original.convert("RGB")
    result = candidate.convert("RGB").resize(base.size, Image.Resampling.LANCZOS)
    outside = ImageOps.invert(mask.convert("L").resize(base.size, Image.Resampling.NEAREST))
    difference = ImageChops.difference(base, result).convert("L")
    histogram = difference.histogram()
    changed = sum(index * count for index, count in enumerate(histogram)) / max(1, sum(histogram))
    outside_pixels = sum(1 for value in outside.getdata() if value > 20)
    return round(changed * outside_pixels / max(1, base.width * base.height * 255), 6)


def create_local_preview(original: Image.Image, mask: Image.Image) -> Image.Image:
    """本地无密钥预览：用于跑通流程，不冒充生成式模型结果。"""
    base = original.convert("RGB")
    soft = base.filter(ImageFilter.GaussianBlur(radius=max(2, min(base.size) // 120)))
    soft = ImageEnhance.Color(soft).enhance(0.85)
    return Image.composite(soft, base, mask.convert("L"))


def _downsample_block(arr: np.ndarray, agg: str) -> np.ndarray:
    """2x2 块降采样；agg='mean' 用于图像，'max' 用于掩膜。"""
    h, w = arr.shape[:2]
    ph, pw = h + (h % 2), w + (w % 2)
    if ph != h or pw != w:
        arr = np.pad(arr, ((0, ph - h), (0, pw - w), (0, 0)) if arr.ndim == 3 else ((0, ph - h), (0, pw - w)), mode="edge")
    out = arr.reshape(ph // 2, 2, pw // 2, 2, *arr.shape[2:])
    return out.mean(axis=(1, 3)) if agg == "mean" else out.max(axis=(1, 3))


def deterministic_inpaint(original: Image.Image, mask: Image.Image, iterations: int = 36, min_level: int = 64) -> Image.Image:
    """确定性邻域修补：多尺度调和插值，遮罩内像素只由边界现有像素向内扩散得到。

    不使用任何生成式模型、不联网、不引入任何新内容，同一输入结果完全可复现。
    算法：先降采样到低分辨率铺出大尺度颜色结构，逐层上采样作为下一层初值，
    每层做少量 Jacobi 松弛（局部均值迭代），已知像素作为固定边界。
    相比单层迭代，多尺度能在更少计算量下把边界颜色传播到区域中心。
    """
    base = np.asarray(original.convert("RGB"), dtype=np.float32)
    m = np.asarray(mask.convert("L").resize(original.size, Image.Resampling.NEAREST)) > 16
    if not m.any():
        return original.convert("RGB").copy()

    # 构建金字塔：[(图像, 掩膜), ...] 由粗到细
    pyr: list[tuple[np.ndarray, np.ndarray]] = []
    img_l, mask_l = base, m
    while min(img_l.shape[:2]) > min_level:
        pyr.append((img_l, mask_l))
        img_l = _downsample_block(img_l, "mean")
        mask_l = _downsample_block(mask_l.astype(np.uint8), "max").astype(bool)
    pyr.append((img_l, mask_l))

    # 最粗层初值取自身，逐层细化
    result = pyr[-1][0].copy()
    for level in range(len(pyr) - 1, -1, -1):
        img_l, mask_l = pyr[level]
        if result.shape != img_l.shape:
            result = np.asarray(
                Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), "RGB").resize(
                    (img_l.shape[1], img_l.shape[0]), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )
        mask3 = np.repeat(mask_l[:, :, None], 3, axis=2)
        result = np.where(mask3, result, img_l)
        for _ in range(max(8, int(iterations))):
            padded = np.pad(result, ((1, 1), (1, 1), (0, 0)), mode="edge")
            neighbor_mean = (
                padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
            ) * 0.25
            result = np.where(mask3, neighbor_mean, img_l)
        result = np.where(mask3, result, img_l)

    out = np.where(np.repeat(m[:, :, None], 3, axis=2), result, base)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def make_demo_mural(size: tuple[int, int] = (960, 640)) -> Image.Image:
    """生成不依赖版权素材的合成演示图，并放置可见的模拟异常。"""
    width, height = size
    image = Image.new("RGB", size, (191, 126, 74))
    draw = ImageDraw.Draw(image)
    for y in range(height):
        ratio = y / max(1, height - 1)
        color = (int(185 + 30 * ratio), int(117 + 30 * ratio), int(69 + 24 * ratio))
        draw.line((0, y, width, y), fill=color)
    draw.rectangle((22, 22, width - 23, height - 23), outline=(75, 50, 40), width=10)
    draw.rectangle((48, 48, width - 49, height - 49), outline=(224, 178, 97), width=5)
    draw.ellipse((120, 128, 290, 298), fill=(225, 185, 87), outline=(83, 49, 37), width=6)
    draw.polygon([(90, 470), (260, 300), (420, 455), (590, 260), (860, 470)], fill=(90, 86, 66))
    draw.line((90, 470, 860, 470), fill=(52, 48, 42), width=10)
    draw.arc((330, 130, 640, 440), start=200, end=340, fill=(74, 43, 36), width=12)
    draw.ellipse((455, 190, 515, 250), fill=(72, 42, 33))
    draw.line((480, 250, 480, 370), fill=(72, 42, 33), width=18)
    draw.line((480, 280, 390, 335), fill=(72, 42, 33), width=12)
    draw.line((480, 280, 570, 335), fill=(72, 42, 33), width=12)
    draw.line((480, 370, 420, 455), fill=(72, 42, 33), width=14)
    draw.line((480, 370, 550, 455), fill=(72, 42, 33), width=14)
    draw.arc((350, 100, 610, 520), start=15, end=165, fill=(220, 165, 86), width=14)
    # 模拟异常1：浅色不规则剥落/裂纹块（与demo_analysis坐标对应）
    draw.polygon([(458, 348), (492, 338), (528, 350), (542, 388), (520, 420), (482, 430), (452, 402)], fill=(224, 180, 133))
    draw.line((472, 360, 501, 378, 487, 410), fill=(130, 76, 58), width=3)
    draw.line((507, 350, 514, 382, 535, 397), fill=(130, 76, 58), width=3)
    # 模拟异常2：右上方暗色表面斑块
    draw.ellipse((672, 224, 816, 326), fill=(110, 77, 58), outline=(77, 52, 45), width=4)
    draw.ellipse((700, 242, 790, 304), fill=(88, 64, 56))
    draw.text((70, height - 92), "壁鉴新生 · synthetic demo mural", fill=(65, 40, 32))
    return image
