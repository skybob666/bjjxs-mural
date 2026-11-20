"""生成单页 A4 观察卡，按区域编号呈现可核对的证据。"""

from datetime import datetime
from io import BytesIO
from pathlib import Path
import os
import re

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont, TTFError
from reportlab.pdfgen import canvas

from image_utils import crop_region
from review_evidence import card_issues, card_recommendations, card_regions, current_review, region_code

WIDTH, HEIGHT = A4
MARGIN = 34
CONTENT = WIDTH - MARGIN * 2
PAPER = colors.HexColor("#FCF8F0")
INK = colors.HexColor("#352D27")
MUTED = colors.HexColor("#6B5B51")
RED = colors.HexColor("#943B2D")
TEAL = colors.HexColor("#1D6265")
LINE = colors.HexColor("#DFD3C0")
TINT = colors.HexColor("#F2EBDD")


def _font():
    name = "MuralCardCJK"
    if name in pdfmetrics.getRegisteredFontNames():
        return name
    fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    try:
        pdfmetrics.registerFont(TTFont(name, str(fonts / "msyh.ttc"), subfontIndex=0))
        return name
    except (OSError, TTFError):
        if "STSong-Light" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        return "STSong-Light"


def _text(pdf, value, x, y, size=9, color=INK):
    pdf.setFont(_font(), size)
    pdf.setFillColor(color)
    pdf.drawString(x, y, str(value))


def _lines(pdf, value, x, y, width, size=9, leading=11, limit=2, color=INK):
    rows = _wrap_rows(value, width, size)
    if len(rows) > limit:
        rows = rows[:limit]
        font = _font()
        while pdfmetrics.stringWidth(rows[-1] + "…", font, size) > width:
            rows[-1] = rows[-1][:-1]
        rows[-1] += "…"
    for row in rows:
        _text(pdf, row, x, y, size, color)
        y -= leading
    return y


def _wrap_rows(value, width, size):
    value = " ".join(str(value or "").split())
    font = _font()
    rows = []
    line = ""
    for ch in value:
        if line and pdfmetrics.stringWidth(line + ch, font, size) > width:
            rows.append(line)
            line = ""
        line += ch
    if line:
        rows.append(line)
    return rows


def _complete_overview(value, width, size=11, max_lines=2):
    """Keep the overview at sentence boundaries so the card never ends mid-claim."""
    normalized = " ".join(str(value or "").split())
    if not normalized:
        return "待补充整体观察。"
    chunks = re.findall(r".*?[。！？；]|.+$", normalized)
    selected = ""
    for chunk in chunks:
        candidate = selected + chunk
        if len(_wrap_rows(candidate, width, size)) <= max_lines:
            selected = candidate
        else:
            break
    return selected or chunks[0]


def _image(pdf, image, x, y, w, h):
    ratio = min(w / image.width, h / image.height)
    iw, ih = image.width * ratio, image.height * ratio
    result = image.convert("RGB")
    result.thumbnail((max(1, round(iw * 3)), max(1, round(ih * 3))), Image.Resampling.LANCZOS)
    buf = BytesIO()
    result.save(buf, format="JPEG", quality=93)
    buf.seek(0)
    pdf.setFillColor(TINT)
    pdf.rect(x, y, w, h, fill=1, stroke=0)
    ix, iy = x + (w - iw) / 2, y + (h - ih) / 2
    pdf.drawImage(ImageReader(buf), ix, iy, iw, ih)
    return ix, iy, iw, ih


def _mark(pdf, region, placement):
    x, y, w, h = placement
    x1, y1, x2, y2 = (value / 1000 for value in region["bbox"])
    bx, by = x + w * x1, y + h * (1 - y2)
    bw, bh = w * (x2 - x1), h * (y2 - y1)
    color = TEAL if current_review(region) else RED
    pdf.setLineWidth(2.6)
    pdf.setStrokeColor(colors.white)
    pdf.rect(bx, by, bw, bh, fill=0)
    pdf.setLineWidth(1.25)
    pdf.setStrokeColor(color)
    pdf.rect(bx, by, bw, bh, fill=0)
    lx = max(x, min(bx, x + w - 29))
    ly = max(y, min(by + bh, y + h - 15))
    pdf.setFillColor(color)
    pdf.rect(lx, ly, 29, 15, fill=1, stroke=0)
    _text(pdf, region_code(region), lx + 3, ly + 3.5, 9.3, colors.white)


def _rule(pdf, y):
    pdf.setLineWidth(0.55)
    pdf.setStrokeColor(LINE)
    pdf.line(MARGIN, y, WIDTH - MARGIN, y)


def _draw_region(pdf, original, region, x, y, width, row_height, compact):
    code = region_code(region)
    review = current_review(region)
    crop_width = 78 if compact else 97
    # Leave a visible breathing gap before the next row.  The old crop height
    # put the image bottom on top of the following row's heading/divider.
    crop_height = min(111, row_height - 36)
    crop = crop_region(original, region["bbox"])
    _image(pdf, crop, x, y - 20 - crop_height, crop_width, crop_height)
    text_x = x + crop_width + 10
    text_width = width - crop_width - 10
    _text(pdf, f"{code}   {'已确认' if review else '待复核'}", x, y, 13, TEAL if review else RED)

    if review:
        cursor = _lines(pdf, "现象  " + review["observation"], text_x, y - 24, text_width, 11.2, 14.1, 3 if compact else 4)
        if review.get("uncertainty"):
            cursor = _lines(pdf, "待核  " + review["uncertainty"], text_x, cursor - 2, text_width, 10.5, 13.1, 3, MUTED)
        return

    original_ai = region.get("source") == "AI候选" and region.get("ai_bbox") == region.get("bbox")
    if original_ai and region.get("visible_observation"):
        _lines(pdf, "AI 观察  " + region["visible_observation"], text_x, y - 24, text_width, 11, 14.2, 4)
    else:
        _lines(pdf, "请对照局部图，确认本框可见现象。", text_x, y - 24, text_width, 11, 14.2, 3, MUTED)


def _advice(pdf, analysis, y):
    _rule(pdf, y + 12)
    _text(pdf, "保护或展示建议", MARGIN, y - 4, 13, RED)
    recommendations = card_recommendations(analysis)
    if not recommendations:
        _text(pdf, "待填写与确认区域对应的建议及风险。", MARGIN, y - 29, 11, MUTED)
        return y - 51

    if len(recommendations) > 2:
        _text(pdf, "草稿预览前两条建议", MARGIN + 271, y - 4, 10, RED)
    point_y = y - 26
    for item in recommendations[:2]:
        _text(pdf, item["id"], MARGIN, point_y, 11.5, TEAL)
        text_x = MARGIN + 38
        cursor = _lines(pdf, item["action"], text_x, point_y, CONTENT - 38, 11.2, 14.3, 2)
        cursor = _lines(
            pdf, f"依据见 {item['id']} 可见现象。风险  " + item["risk"],
            text_x, cursor - 2, CONTENT - 38, 10.5, 13.5, 3, MUTED,
        )
        point_y = cursor - 6
    return point_y


def build_observation_card_pdf(task, original, annotated, analysis, audit_log=None):
    """单页 A4，左边全貌定位，右边同编号的局部证据。"""
    output = BytesIO()
    pdf = canvas.Canvas(output, pagesize=A4, pageCompression=1)
    pdf.setTitle(str(task.get("title") or "壁鉴新生 · AI 壁画观察卡"))
    pdf.setFillColor(PAPER)
    pdf.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
    pdf.setFillColor(RED)
    pdf.rect(0, HEIGHT - 62, WIDTH, 62, fill=1, stroke=0)
    pdf.setFillColor(colors.HexColor("#E2BF74"))
    pdf.rect(0, HEIGHT - 65, WIDTH, 3, fill=1, stroke=0)
    _text(pdf, "壁鉴新生  /  AI 壁画观察卡", MARGIN, HEIGHT - 40, 22, colors.white)
    issues = card_issues(analysis, task)
    mode = str(task.get("analysis_mode") or analysis.get("analysis_mode") or "未说明")
    card_type = "练习样例" if "回放" in mode else ("观察草稿" if issues else "图像观察记录")
    pdf.setFont(_font(), 10)
    pdf.setFillColor(colors.white)
    pdf.drawRightString(WIDTH - MARGIN, HEIGHT - 39, card_type)
    _lines(pdf, task.get("title", "壁画局部图像观察"), MARGIN, 750, CONTENT - 125, 15, limit=1)
    _lines(pdf, task.get("task_id", "待编号"), WIDTH - 144, 751, 110, 10, limit=1, color=TEAL)

    is_ai = "DeepSeek" in mode and "兜底" not in mode
    _text(pdf, "AI 观察" if is_ai else "全图初览", MARGIN, 718, 12, RED)
    overview_width = CONTENT - 64
    overall = _complete_overview(analysis.get("overall_observation"), overview_width)
    _lines(pdf, overall, MARGIN + 64, 718, overview_width, 11, 14.3, 2)
    _rule(pdf, 686)

    left_width = 186
    right_x = MARGIN + left_width + 17
    right_width = WIDTH - MARGIN - right_x
    _text(pdf, "原图全貌", MARGIN, 669, 13, RED)
    _text(pdf, "局部证据与复核", right_x, 669, 13, RED)
    regions = card_regions(analysis)
    band_top = 646
    row_height = {1: 177, 2: 136, 3: 116}.get(len(regions), 190)
    band_bottom = band_top - row_height * (len(regions) if 1 <= len(regions) <= 3 else 1)
    large_preview = annotated if len(regions) > 3 else original
    placement = _image(pdf, large_preview, MARGIN, band_bottom + 15, left_width, band_top - band_bottom - 7)
    if 1 <= len(regions) <= 3:
        for index, region in enumerate(regions):
            _mark(pdf, region, placement)
            row_y = band_top - index * row_height
            _draw_region(pdf, original, region, right_x, row_y, right_width, row_height, len(regions) == 3)
    elif regions:
        _lines(pdf, "内部草稿。区域编号：" + "　".join(map(region_code, regions)), right_x, band_top, right_width, 11, 14, 11)
    else:
        _text(pdf, "暂无可展示的区域。", right_x, band_top, 11, MUTED)

    advice_end = _advice(pdf, analysis, band_bottom - 12)
    rejected = [r for r in analysis.get("regions", []) if r.get("status") == "已驳回" and current_review(r)]
    if rejected and advice_end > 167:
        first = rejected[0]
        _lines(pdf, f"驳回 {region_code(first)}  " + current_review(first)["reason"], MARGIN, advice_end - 6, CONTENT, 10.5, 13, 2, MUTED)

    _rule(pdf, 132)
    source_y = _lines(pdf, "图像来源  " + str(task.get("image_source") or "待补充"), MARGIN, 115, CONTENT, 10, 12, 2)
    source_y = _lines(pdf, "使用说明  " + str(task.get("authorization") or "待补充"), MARGIN, source_y, CONTENT, 10, 12, 2)
    source_y = _lines(pdf, "观察工具  " + mode + "  /  壁鉴新生", MARGIN, source_y, CONTENT, 10, 12, 1)
    ref = "术语参考  GB/T 30237-2025《古代壁画病害与图示》"
    _text(pdf, ref, MARGIN, source_y, 9.8, TEAL)
    pdf.linkURL("https://std.samr.gov.cn/gb/search/gbDetailed?id=31DA5F377C128F08E06397BE0A0A4CFB",
                (MARGIN, source_y - 2, MARGIN + pdfmetrics.stringWidth(ref, _font(), 9.8), source_y + 10))
    reviewed = sum(bool(current_review(r)) for r in regions)
    trace = f"已确认 {reviewed} 处  ·  {datetime.now():%Y-%m-%d}"
    if rejected:
        trace += "  ·  已驳回 " + "、".join(map(region_code, rejected))
    _lines(pdf, trace, MARGIN, source_y - 12, CONTENT, 9.8, 12, 1, MUTED)
    _text(pdf, "确认范围为图上可见现象。病害性质与成因需结合原件核查。", MARGIN, 25, 10, MUTED)
    pdf.showPage()
    pdf.save()
    return output.getvalue()
