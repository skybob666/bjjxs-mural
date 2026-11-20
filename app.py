"""壁鉴新生：壁画智能观察、人工复核、观察卡与修复实验工作台。"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image, ImageFile, ImageOps

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(dotenv_path: str | None = None) -> bool:
        path = Path(dotenv_path or Path(__file__).resolve().parent / ".env")
        if not path.exists():
            return False
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        return True

try:
    from streamlit_drawable_canvas import st_canvas
except Exception as exc:
    st_canvas = None
    _canvas_import_error = str(exc)

from ai_service import analyze_mural, demo_analysis, fallback_analysis, test_deepseek_connection
from image_utils import (
    crop_region,
    canvas_delta_to_bbox,
    canvas_size_for_image,
    draw_regions,
    fabric_rectangles_to_bboxes,
    make_demo_mural,
    validate_bbox,
)
from local_secrets import remove_env_values, save_env_values
from observation_card import build_observation_card_pdf
from report import build_report, export_json, image_bytes, save_outputs
from review_evidence import card_issues, current_review, region_code, review_at_bbox, save_region_review

load_dotenv()
ImageFile.LOAD_TRUNCATED_IMAGES = True


def apply_streamlit_secrets() -> bool:
    """云端部署时把 Streamlit Secrets 注入环境变量；本地 .env 优先，不会被覆盖。"""
    try:
        items = {k: v for k, v in st.secrets.items() if isinstance(v, str)}
    except Exception:
        return False
    for key, value in items.items():
        os.environ.setdefault(key, value)
    return bool(items)

BASE_DIR = Path(__file__).resolve().parent
HERO_PATH = BASE_DIR / "assets" / "dunhuang_hero_v1.png"
ONLINE_MODE = "线上研发 / 校赛演示"
COMPLIANCE_MODE = "现场合规预览"
LOCAL_MODE = "本地演示（无外部API）"

st.set_page_config(page_title="壁鉴新生", page_icon="🏛️", layout="wide", initial_sidebar_state="expanded")
CLOUD_SECRETS_LOADED = apply_streamlit_secrets()


def apply_style() -> None:
    st.markdown(
        """
        <style>
        :root { --paper:#fcf8f0; --ink:#3e342d; --cinnabar:#a13b2b; --turq:#1e777d; --gold:#c6953c; }
        .stApp { background: linear-gradient(135deg,#fcf8f0 0%,#f6eee0 100%); color:var(--ink); }
        [data-testid="stSidebar"] { background:#f0e6d4; border-right:1px solid #d8c3a0; }
        .hero-copy { margin-top:-7.5rem; padding:2.4rem 2.4rem 2rem; color:white; text-shadow:0 2px 12px rgba(30,20,12,.48); background:linear-gradient(90deg,rgba(62,52,45,.78),rgba(62,52,45,.20)); border-radius:0 0 14px 14px; position:relative; }
        .hero-copy h1 { font-size:3.1rem; letter-spacing:.08em; margin:0 0 .35rem; }
        .hero-copy p { font-size:1.08rem; margin:0; opacity:.94; }
        .section-chip { display:inline-block; color:#fff; background:var(--cinnabar); border-radius:999px; padding:.25rem .75rem; font-size:.82rem; font-weight:700; }
        .metric-card { background:rgba(255,255,255,.65); border:1px solid #ddcdb4; border-radius:14px; padding:1rem 1.1rem; min-height:88px; }
        .metric-label { color:#75685c; font-size:.78rem; }
        .metric-value { color:var(--cinnabar); font-size:1.55rem; font-weight:750; margin-top:.3rem; }
        .boundary { border-left:4px solid var(--gold); background:#f5ead4; padding:.75rem 1rem; border-radius:0 10px 10px 0; }
        .review-heading { font-size:1.02rem; font-weight:750; color:var(--ink); margin:.05rem 0 .15rem; }
        .review-step { color:#75685c; font-size:.82rem; margin-bottom:.45rem; }
        .coord-note { color:#75685c; font-size:.78rem; line-height:1.55; }
        div[data-testid="stTabs"] button { color:#75685c; }
        div[data-testid="stTabs"] button[aria-selected="true"] { color:var(--cinnabar); font-weight:750; }
        @media (max-width: 600px) {
            .hero-copy { margin-top:-6.2rem; padding:1.7rem 1rem 1.2rem; }
            .hero-copy h1 { font-size:2rem; line-height:1.05; letter-spacing:.04em; }
            .hero-copy p { font-size:.9rem; line-height:1.45; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def add_audit(action: str, detail: str) -> None:
    st.session_state.setdefault("audit_log", []).append(
        {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "action": action, "detail": detail}
    )


STEP_LABELS = ["01 建立档案", "02 AI观察", "03 人工复核", "04 观察卡"]
# 依据 GB/T 30237-2025 和敦煌研究院、Getty 莫高窟考察，按图像可见特征组织。
REVIEW_PRESETS = {
    "observation": (
        "裂纹或裂隙：可见条状或网状缝线。",
        "脱落或露底：着色区域中断，局部可见底色。",
        "翘起或鼓起迹象：可见片缘卷曲或局部隆起。",
        "粉状或颗粒状附着：表面可见浅色细粒或粉状物。",
        "色差或污渍状斑：局部偏浅、偏深或有异常斑块。",
    ),
    "rejected_reason": (
        "框的位置与描述不对应。",
        "可见线条或色块与原有纹样连续。",
        "疑似反光、阴影或图像伪影。",
        "局部看不清，证据不足。",
        "与其他候选区域重复。",
    ),
    "uncertainty": (
        "现有图像无法区分原有纹样与表面变化。",
        "需比较不同光照的图像，核查反光或阴影影响。",
        "需核对高清局部，排除扫描或压缩影响。",
        "需用侧光核查是否存在翘起或隆起。",
        "粉状物的成分及附着状态需现场核查。",
    ),
    "action": (
        "按同一位置补拍高清局部并建立对照记录。",
        "收集正光与侧光图像，交由专业人员核对。",
        "对比同位置的历史图像，记录范围有无变化。",
        "数字展示时用可关闭的标记指示该处并注明待核。",
    ),
    "risk": (
        "拍摄角度或光照不同，可能造成对照偏差。",
        "相机或扫描色彩变化，可能放大图像差异。",
        "拍摄条件不一致会影响对变化范围的判断。",
        "标记若遮住原有纹样，可能误导观众。",
    ),
}
OTHER_REVIEW_TEXT = "其他（手动填写）"


def review_text_choice(label: str, preset: str, value: str, key: str, limit: int) -> str:
    choices = [*REVIEW_PRESETS[preset], OTHER_REVIEW_TEXT]
    selected = value if value in choices else (OTHER_REVIEW_TEXT if value else None)
    choice = st.selectbox(
        label, choices, index=choices.index(selected) if selected else None,
        key=f"{key}_choice", placeholder="选择常用表述或选「其他」",
    )
    if choice == OTHER_REVIEW_TEXT:
        return st.text_area(
            f"其他：{label}", value=value if value not in choices else "",
            key=f"{key}_other", max_chars=limit, height=80,
        )
    return choice or ""


def scroll_anchor(name: str, jump: bool) -> None:
    st.html(f'<div id="{name}" style="height:1px;scroll-margin-top:4.5rem"></div>')
    if jump:
        script = f"""
        <script>
        {{
          const started = performance.now();
          const move = () => {{
            const app = document.querySelector("[data-test-script-state]");
            if (app && app.getAttribute("data-test-script-state") !== "notRunning"
                && performance.now() - started < 900) {{
              setTimeout(move, 60);
              return;
            }}
            document.getElementById({json.dumps(name)})
              ?.scrollIntoView({{block: "start", behavior: "instant"}});
          }};
          requestAnimationFrame(() => requestAnimationFrame(move));
        }}
        </script>
        """
        try:
            st.html(script, unsafe_allow_javascript=True)
        except TypeError:
            from streamlit.components.v1 import html

            html(script.replace("document.", "parent.document."), height=0)


def request_step(target: int, message: str | None = None) -> None:
    """登记跳转请求到独立状态键，重跑后在步骤条控件创建前消费。

    active_step 是步骤条 radio 的绑定 key，不能在控件实例化后被修改；
    因此跳转请求先写入非控件的 pending_step，由 consume_flow_navigation() 落地。
    """
    st.session_state["pending_step"] = int(target)
    if message:
        st.session_state["flow_toast"] = message
    st.rerun()


def consume_flow_navigation() -> None:
    """在步骤条创建前消费跳转请求并落地到 active_step，随后显示流程提示。"""
    pending = st.session_state.pop("pending_step", None)
    if pending is not None and 0 <= int(pending) < len(STEP_LABELS):
        st.session_state["active_step"] = STEP_LABELS[int(pending)]
    toast = st.session_state.pop("flow_toast", None)
    if toast:
        try:
            st.toast(toast)
        except Exception:
            st.success(toast)


# 建档字段默认来源信息：区分真实上传图与内置合成演示图，避免来源披露不实。
DEFAULT_DEMO_SOURCE = "团队自制合成演示图（非真实文物图像，仅用于流程演练）"
DEFAULT_DEMO_AUTHORIZATION = "内置合成演示图，由本项目代码生成，不涉及第三方素材与文物图像授权。"
DEFAULT_TITLE_REAL = "壁画局部图像观察"
DEFAULT_TITLE_DEMO = "合成演示壁画局部保存状况观察"


def reset_for_new_image(signature: str, *, is_demo: bool = False) -> None:
    if st.session_state.get("image_signature") == signature:
        return
    for key in ("analysis", "selected_region_id", "pending_review_region_id", "review_flash", "audit_log"):
        st.session_state.pop(key, None)
    for key in list(st.session_state):
        if key.startswith("candidate_canvas_snapshot_"):
            st.session_state.pop(key, None)
    # 换图时按图像类型重置建档字段默认值；用户随后可手动修改。
    st.session_state["task_title"] = DEFAULT_TITLE_DEMO if is_demo else DEFAULT_TITLE_REAL
    st.session_state["image_source"] = DEFAULT_DEMO_SOURCE if is_demo else ""
    st.session_state["authorization"] = DEFAULT_DEMO_AUTHORIZATION if is_demo else ""
    st.session_state.setdefault("task_id", "BJ-2026-001")
    st.session_state["image_signature"] = signature
    add_audit("建立新图像档案", f"图像指纹 {signature[:12]}")


def region_label(region: dict[str, Any]) -> str:
    return f"R{region.get('id', '?')} · {region.get('suspected_issue', '未知异常')} · {region.get('status', '待复核')}"


def original_ai_bbox(region: dict[str, Any]) -> list[int] | None:
    """返回可追溯的AI原始框；人工新增和本地兜底不冒充AI结果。"""
    raw_bbox = region.get("ai_bbox")
    if raw_bbox is None and "ai_bbox" not in region and region.get("source") in {"AI候选", "本地演示"}:
        raw_bbox = region.get("bbox")
    return validate_bbox(raw_bbox)


def ai_reference_regions(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for region in regions:
        bbox = original_ai_bbox(region)
        if bbox is None:
            continue
        references.append({**region, "bbox": bbox, "status": "AI原始", "human_reviewed": False})
    return references


def render_metrics(analysis: dict[str, Any] | None) -> None:
    regions = (analysis or {}).get("regions", [])
    reviewed = sum(1 for item in regions if item.get("status") in {"已确认", "已驳回"})
    cols = st.columns(4)
    values = [("候选区域", len(regions)), ("已人工复核", reviewed), ("待复核", max(0, len(regions) - reviewed)), ("过程记录", len(st.session_state.get("audit_log", [])))]
    for col, (label, value) in zip(cols, values):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)


def make_task(image_name: str, image_source: str, title: str, task_id: str, authorization: str, analysis_mode: str) -> dict[str, Any]:
    return {
        "task_id": task_id or "BJ-未编号",
        "title": title or "未命名壁画观察任务",
        "image_name": image_name,
        "image_source": image_source or "待补充",
        "authorization": authorization or "待补充",
        "analysis_mode": analysis_mode,
    }


def load_uploaded_image(source_bytes: bytes) -> Image.Image:
    """兼容常见图片格式、EXIF方向和轻微截断文件，并统一为RGB。"""
    try:
        with Image.open(BytesIO(source_bytes)) as opened:
            try:
                corrected = ImageOps.exif_transpose(opened)
            except Exception:
                corrected = opened
            corrected.load()
            return corrected.convert("RGB")
    except Exception as exc:
        raise ValueError(f"图片无法读取或格式已损坏：{exc}") from exc


def render_manual_canvas(image: Image.Image, signature: str, analysis: dict[str, Any]) -> None:
    st.markdown('<span class="section-chip">AI标注与人工补画对照</span>', unsafe_allow_html=True)
    st.caption("左侧只保留AI最初提出的框，右侧用于人工补画；两张图尺寸一致，方便逐处比较。")
    canvas_size = canvas_size_for_image(image, max_width=520, max_height=620)
    background = image.resize(canvas_size, Image.Resampling.LANCZOS)
    references = ai_reference_regions(analysis.get("regions", []))
    ai_preview = draw_regions(image, references).resize(canvas_size, Image.Resampling.LANCZOS)

    ai_col, manual_col = st.columns(2, gap="large")
    with ai_col:
        with st.container(border=True):
            st.markdown('<div class="review-heading">左 · AI原始候选</div>', unsafe_allow_html=True)
            st.markdown('<div class="review-step">查看AI最初标在哪里，不随人工修改而改变</div>', unsafe_allow_html=True)
            st.image(ai_preview if references else background, width=canvas_size[0])
            if references:
                st.caption(f"共 {len(references)} 个AI原始候选。这里只用于对照，不能在此直接改框。")
            else:
                st.info("本次没有可追溯的AI原始框。可在右侧人工新增候选。")

    with manual_col:
        with st.container(border=True):
            st.markdown('<div class="review-heading">右 · 人工新增候选</div>', unsafe_allow_html=True)
            st.markdown('<div class="review-step">在图上拖出矩形，再把最后一个矩形加入候选</div>', unsafe_allow_html=True)
            if st_canvas is None:
                canvas = None
                st.image(background, width=canvas_size[0])
                st.info("当前环境没有加载画布组件，可在下方用坐标精确调整现有区域。")
            else:
                canvas = st_canvas(
                    fill_color="rgba(161, 59, 43, 0.18)",
                    stroke_width=3,
                    stroke_color="#A13B2B",
                    background_image=background,
                    update_streamlit=True,
                    return_image_data=True,
                    height=canvas_size[1],
                    width=canvas_size[0],
                    drawing_mode="rect",
                    key=f"candidate_canvas_{signature[:12]}",
                )
            snapshot_key = f"candidate_canvas_snapshot_{signature[:12]}"
            snapshot_image_key = f"{snapshot_key}_image"
            boxes: list[list[int]] = []
            detection_note = ""
            if canvas is not None:
                live_json = getattr(canvas, "json_data", None)
                live_image = getattr(canvas, "image_data", None)
                live_boxes = fabric_rectangles_to_bboxes(live_json, canvas_size, image.size)
                if live_boxes:
                    st.session_state[snapshot_key] = live_json
                    boxes = live_boxes
                    detection_note = "已读取画布矩形"
                else:
                    live_bbox = canvas_delta_to_bbox(live_image, background, canvas_size, image.size)
                    if live_bbox is not None:
                        st.session_state[snapshot_image_key] = live_image
                        boxes = [live_bbox]
                        detection_note = "已从画布标记恢复矩形"
                    else:
                        saved_boxes = fabric_rectangles_to_bboxes(st.session_state.get(snapshot_key), canvas_size, image.size)
                        if saved_boxes:
                            boxes = saved_boxes
                            detection_note = "已保留刚才画出的矩形"
                        else:
                            saved_bbox = canvas_delta_to_bbox(st.session_state.get(snapshot_image_key), background, canvas_size, image.size)
                            if saved_bbox is not None:
                                boxes = [saved_bbox]
                                detection_note = "已保留刚才画出的标记"
            add_clicked = st.button(
                "＋ 将最后一个矩形加入候选",
                key=f"add_canvas_region_{signature[:12]}",
                width="stretch",
                disabled=canvas is None,
                help="若画了多个框，只会加入最后画出的一个。",
            )
            st.caption("操作提示：需要重画时，选中矩形后按 Delete；加入后在下方填写人工结论。")
            if boxes:
                st.success(f"{detection_note}：可加入 {len(boxes)} 个矩形。点击上方按钮会加入最后一个。")

            if add_clicked and canvas is not None:
                if not boxes:
                    st.warning("没有读到有效矩形。请先画框，松开鼠标后确认出现“可加入矩形”提示，再点击加入。")
                else:
                    bbox = boxes[-1]
                    existing = {tuple(item.get("bbox") or []) for item in analysis.get("regions", [])}
                    if tuple(bbox) in existing:
                        st.warning("这个区域已在候选列表中。请重画一个不同的位置，或在下方直接复核已有区域。")
                        return
                    new_id = max([int(item.get("id", 0)) for item in analysis.get("regions", [])] + [0]) + 1
                    analysis["regions"].append(
                        {
                            "id": new_id,
                            "bbox": bbox,
                            "ai_bbox": None,
                            "visible_observation": "人工在图像上标记的待观察区域。",
                            "suspected_issue": "人工新增观察候选",
                            "evidence": "由人工依据当前图像位置和纹理差异提出。",
                            "uncertainty": "需要结合原始图像、现场信息和专业人员进一步确认。",
                            "recommendation": "保留标注位置并补充拍摄条件或现场检测记录。",
                            "source": "人工框选",
                            "status": "待复核",
                            "human_note": "",
                            "human_review_required": True,
                            "human_reviewed": False,
                        }
                    )
                    st.session_state["analysis"] = analysis
                    st.session_state["pending_review_region_id"] = new_id
                    add_audit("人工新增候选", f"R{new_id}，坐标 {bbox}；{detection_note}")
                    st.session_state["review_flash"] = f"已加入 R{new_id:02d}，可以开始复核。"
                    st.rerun()


def render_analysis(image: Image.Image, image_name: str, analysis_mode: str, deepseek_key: str, deepseek_model: str, is_synthetic: bool) -> None:
    st.markdown('<span class="section-chip">AI提出候选，不替代专业鉴定</span>', unsafe_allow_html=True)
    st.write("用结构化观察记录可见现象、疑似问题、证据与不确定性，避免把模型猜测当成结论。")
    if analysis_mode == COMPLIANCE_MODE:
        st.warning("现场合规预览已锁定外部API：本模式只使用本地演示/人工流程。比赛现场请按规则使用千问办公与百炼。")
    if st.button("开始一次结构化观察", type="primary", width="stretch"):
        try:
            with st.spinner("正在形成候选区域与观察依据…"):
                if analysis_mode == ONLINE_MODE:
                    result = analyze_mural(image, deepseek_key, deepseek_model)
                    result["analysis_mode"] = f"DeepSeek · {deepseek_model}"
                else:
                    result = demo_analysis(image, synthetic=is_synthetic)
                    result["analysis_mode"] = "本地演示数据（未调用外部模型）"
            st.session_state["analysis"] = result
            st.session_state.pop("selected_region_id", None)
            add_audit("完成AI观察", result["analysis_mode"])
            request_step(2, "AI观察完成，已自动进入人工复核。请逐区域确认、修改或驳回。")
        except Exception as exc:
            if analysis_mode == ONLINE_MODE:
                result = fallback_analysis(image, str(exc))
                st.session_state["analysis"] = result
                st.session_state.pop("selected_region_id", None)
                add_audit("DeepSeek失败，切换本地兜底", str(exc)[:500])
                request_step(2, "DeepSeek未完成，已切换本地兜底并进入人工复核；结果不是AI识别结论。")
            else:
                st.error(f"观察失败：{exc}")
    analysis = st.session_state.get("analysis")
    if not analysis:
        st.info("点击上方按钮建立候选区域，或先去‘03 人工复核’使用人工选区。")
        return
    render_metrics(analysis)
    st.subheader("整体观察")
    st.write(analysis.get("overall_observation", ""))
    annotated = draw_regions(image, analysis.get("regions", []))
    st.image(annotated, caption="颜色框：AI/演示候选；金色框：已人工确认", width="stretch")
    for region in analysis.get("regions", []):
        with st.expander(region_label(region), expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**直接可见现象**：{region.get('visible_observation', '')}")
                st.write(f"**判断依据**：{region.get('evidence', '')}")
            with c2:
                st.write(f"**疑似问题**：{region.get('suspected_issue', '')}")
                st.write(f"**不确定性**：{region.get('uncertainty', '')}")
            st.caption(f"来源：{region.get('source', 'AI候选')} · 坐标：{region.get('bbox', [])} · 状态：{region.get('status', '待复核')}")


def render_review(image: Image.Image, signature: str) -> Any | None:
    analysis = st.session_state.get("analysis")
    if not analysis:
        st.info("请先完成‘02 AI观察’，再进入人工复核。")
        return None
    regions = analysis.get("regions", [])
    for item in regions:
        if "ai_bbox" not in item:
            item["ai_bbox"] = list(item.get("bbox", [])) if item.get("source") in {"AI候选", "本地演示"} else None
    if not regions:
        st.info("当前没有候选区域。可以在下方标记需要观察的区域。")
        render_manual_canvas(image, signature, analysis)
        return None
    ids = [int(item.get("id", 0)) for item in regions]
    selector_key = f"review_region_{signature[:12]}"
    pending_id = st.session_state.pop("pending_review_region_id", None)
    if pending_id in ids:
        st.session_state[selector_key] = ids.index(pending_id)
    if selector_key not in st.session_state or st.session_state[selector_key] not in range(len(regions)):
        first_pending = next((index for index, item in enumerate(regions) if not current_review(item)), 0)
        previous = st.session_state.get("selected_region_id")
        st.session_state[selector_key] = ids.index(previous) if previous in ids else first_pending
    review_flash = st.session_state.pop("review_flash", None)
    scroll_anchor("current-review", jump=bool(review_flash))
    st.markdown("### 逐区域人工复核")
    completed = sum(bool(current_review(item)) for item in regions)
    st.caption(f"已保存 {completed}/{len(regions)} 处。选择常用表述，或用「其他」写入本图的细节。")
    selected_index = st.selectbox(
        "选择需要确认的区域", list(range(len(regions))), key=selector_key,
        format_func=lambda index: f"{region_code(regions[index])} · {'已保存' if current_review(regions[index]) else '待填写'}",
    )
    region = regions[selected_index]
    st.session_state["selected_region_id"] = region["id"]
    if review_flash:
        st.success(review_flash)

    code = region_code(region)
    ai_bbox = original_ai_bbox(region)
    prefix = f"review_{signature[:10]}_{region['id']}"
    with st.expander("调整当前框的位置（可选）"):
        st.caption("修改边界后，下方会立即刷新局部图。定位完成后再填写复核。")
        coords = st.columns(4)
        bbox = validate_bbox([
            column.number_input(label, 0, 1000, int(value), step=1, key=f"{prefix}_{index}")
            for index, (column, label, value) in enumerate(zip(
                coords, ("左 X1", "上 Y1", "右 X2", "下 Y2"), region["bbox"],
            ))
        ])
    saved = review_at_bbox(region) if bbox == region["bbox"] else {}

    ai_detail_col, human_detail_col = st.columns(2, gap="large")
    with ai_detail_col:
        with st.container(border=True):
            st.markdown('<div class="review-heading">全貌与区域位置</div>', unsafe_allow_html=True)
            whole = ImageOps.contain(image, (540, 355), Image.Resampling.LANCZOS)
            if bbox:
                whole = draw_regions(whole, [{**region, "bbox": bbox, "status": "当前区域"}], number_only=True)
            st.image(whole, caption=f"{code} 当前框所在位置", width="content")
            if ai_bbox is None:
                st.info("这是人工新增或本地兜底候选，没有AI原始判断。")
            else:
                with st.expander("查看 AI 原始判断"):
                    st.caption(f"来源：{region.get('source', 'AI候选')} · 原框：{ai_bbox}")
                    st.write(region.get("visible_observation") or "未提供可见现象")
                    st.write(region.get("suspected_issue") or "未提供疑似问题")
                    st.write(region.get("uncertainty") or "需结合原图复核")

    with human_detail_col:
        with st.container(border=True):
            st.markdown('<div class="review-heading">局部细节与复核</div>', unsafe_allow_html=True)
            if bbox:
                local = ImageOps.contain(crop_region(image, bbox), (520, 355), Image.Resampling.LANCZOS)
                st.image(local, caption=f"{code} 当前框内原图", width="content")
            else:
                st.warning("当前边界无效，请先调整到能够显示局部图。")
            if bbox != region["bbox"]:
                st.info("框的位置已变化，请按这张局部图重新填写证据。")
            with st.container():
                draft_key = f"{prefix}_{'_'.join(map(str, bbox or []))}"
                statuses = ["已确认", "已驳回"]
                current_status = region.get("status") if saved else None
                status = st.radio(
                    "人工结论",
                    statuses,
                    index=statuses.index(current_status) if current_status in statuses else None,
                    horizontal=True,
                    key=f"{draft_key}_decision",
                    format_func={"已确认": "确认", "已驳回": "驳回"}.get,
                )
                observation = reason = uncertainty = action = risk = ""
                if status == "已驳回":
                    reason = review_text_choice("驳回理由", "rejected_reason", saved.get("reason", ""), f"{draft_key}_reject", 60)
                elif status == "已确认":
                    st.caption("确认的是本框可见现象，成因仍需专业核查。")
                    observation = review_text_choice("框内可见现象", "observation", saved.get("observation", ""), f"{draft_key}_obs", 54)
                    with st.expander("补充待核点或保护建议（可选）", expanded=bool(saved.get("action") or saved.get("uncertainty"))):
                        uncertainty = review_text_choice("待核点（可选）", "uncertainty", saved.get("uncertainty", ""), f"{draft_key}_uncertain", 42)
                        st.caption("整张观察卡留 1–2 条建议。选择建议后请配对填写风险。")
                        action = review_text_choice("保护或数字展示建议（可选）", "action", saved.get("action", ""), f"{draft_key}_action", 52)
                        if action.strip():
                            risk = review_text_choice("建议的可能风险", "risk", saved.get("risk", ""), f"{draft_key}_risk", 52)
                remaining = regions[selected_index + 1:] + regions[:selected_index]
                next_region = next((item for item in remaining if not current_review(item)), None)
                button_label = "保存并复核下一处" if next_region else "保存并查看观察卡"
                if st.button(button_label, type="primary", width="stretch", disabled=bbox is None or status is None, key=f"{draft_key}_save"):
                    before = region["bbox"]
                    try:
                        save_region_review(region, bbox, status, observation, uncertainty, action, risk, rejection_reason=reason)
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        region["human_review_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state["analysis"] = analysis
                        detail = f"驳回理由：{reason}" if reason else f"可见现象：{observation}"
                        add_audit("保存人工复核", f"{code} {status}，{before} → {bbox}，{detail}")
                        if next_region is None:
                            st.session_state["review_flash"] = f"{code} 已保存。所有候选区域均已完成图像复核。"
                            request_step(3)
                        else:
                            st.session_state["pending_review_region_id"] = next_region["id"]
                            st.session_state["review_flash"] = f"{code} 已保存，当前为 {region_code(next_region)}。"
                            st.rerun()
    if st.toggle("查看整图或新增人工候选", key=f"show_candidate_canvas_{signature[:12]}"):
        render_manual_canvas(image, signature, analysis)
    return region


def main() -> None:
    apply_style()
    if HERO_PATH.exists():
        st.image(str(HERO_PATH), width="stretch")
        st.markdown('<div class="hero-copy"><h1>壁鉴新生</h1><p>AI 辅助壁画图像观察与人工复核</p></div>', unsafe_allow_html=True)
    else:
        st.title("🏛️ 壁鉴新生")
    st.caption("壁画智能观察 · 人工复核 · 一页观察卡")

    with st.sidebar:
        st.header("工作模式")
        analysis_mode = st.radio("选择当前环境", [ONLINE_MODE, COMPLIANCE_MODE, LOCAL_MODE], index=0)
        if analysis_mode == COMPLIANCE_MODE:
            st.warning("现场合规预览：外部API已禁用，仅用于演示流程。")
        if st.session_state.pop("clear_deepseek_input", False):
            st.session_state["deepseek_key_input"] = ""
        deepseek_key = st.text_input("DeepSeek API Key", value=os.getenv("DEEPSEEK_API_KEY", ""), type="password", key="deepseek_key_input", disabled=analysis_mode != ONLINE_MODE)
        deepseek_model = st.text_input("DeepSeek 模型", value=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"), key="deepseek_model_input", disabled=analysis_mode != ONLINE_MODE)
        if analysis_mode == ONLINE_MODE:
            if st.button("测试 DeepSeek 连接", disabled=not deepseek_key.strip(), width="stretch"):
                try:
                    with st.spinner("正在检查 Key、网络与可用模型…"):
                        connection = test_deepseek_connection(deepseek_key)
                    models = connection.get("models", [])
                    elapsed = connection.get("elapsed_seconds", "—")
                    if deepseek_model.strip() and deepseek_model.strip() not in models:
                        st.warning(f"连接正常（{elapsed} 秒），但当前填写的模型“{deepseek_model.strip()}”不在可用列表中。")
                    else:
                        st.success(f"DeepSeek 连接正常 · {deepseek_model.strip() or '默认模型'}可用 · {elapsed} 秒")
                except Exception as exc:
                    st.error(f"DeepSeek 连接测试未通过：{exc}")
        if st.button("保存 DeepSeek 配置到本机", disabled=not deepseek_key.strip(), width="stretch"):
            try:
                env_path = save_env_values({"DEEPSEEK_API_KEY": deepseek_key.strip(), "DEEPSEEK_MODEL": deepseek_model.strip() or "deepseek-flash"})
                st.success(f"已保存到本机 .env：{env_path.name}")
            except OSError as exc:
                st.warning(f"当前环境无法写入 .env（{exc}）。云端部署请改用 Secrets 配置。")
        if st.button("清除本机保存的 DeepSeek Key", width="stretch"):
            try:
                env_path = remove_env_values(["DEEPSEEK_API_KEY"])
            except OSError as exc:
                st.warning(f"当前环境无法写入 .env（{exc}）。")
            st.session_state["clear_deepseek_input"] = True
            st.rerun()
        if CLOUD_SECRETS_LOADED:
            st.caption("当前已从云端 Secrets 读取配置；Key不会写入代码，也不会出现在页面源码中。")
        else:
            st.caption("Key只保存在本机项目 .env，不会写入代码；提交压缩包前不要包含 .env。")
        st.divider()
        st.caption("线上阶段仅提交观察卡与演示视频；本工作台不做虚拟修复，现场阶段按赛题要求使用指定平台。")

    uploaded_file = st.file_uploader("上传已获授权的 JPG、PNG、WebP、GIF、BMP 或 TIFF 图像", type=["jpg", "jpeg", "png", "webp", "gif", "bmp", "tif", "tiff"])
    use_demo = st.checkbox("使用内置合成演示图（推荐用于第一次演练）", value=uploaded_file is None)
    if uploaded_file is None and not use_demo:
        st.info("请上传图像或勾选内置演示图。")
        return
    if uploaded_file is not None and not use_demo:
        source_bytes = uploaded_file.getvalue()
        image_name = uploaded_file.name
        try:
            image = load_uploaded_image(source_bytes)
        except ValueError as exc:
            st.error(str(exc))
            return
    else:
        image_name = "synthetic_demo_mural.png"
        image = make_demo_mural()
        source_bytes = image_bytes(image)
    download_bytes = image_bytes(image)
    signature = hashlib.sha256(source_bytes).hexdigest()
    reset_for_new_image(signature, is_demo=use_demo)

    st.info("建议按以下顺序完成一次展示：建立档案 → AI观察 → 人工复核 → 观察卡。")
    consume_flow_navigation()

    # 归一化历史遗留的 int 值；active_step 是唯一状态源。
    raw_step = st.session_state.get("active_step", STEP_LABELS[0])
    if isinstance(raw_step, int):
        step_value = STEP_LABELS[raw_step] if 0 <= raw_step < len(STEP_LABELS) else STEP_LABELS[0]
    else:
        step_value = raw_step if raw_step in STEP_LABELS else STEP_LABELS[0]
    st.session_state["active_step"] = step_value
    # radio 直接绑定 active_step 作为 key：控件显示值与跳转状态是同一个变量，
    # 不存在“控件记住旧选择覆盖会话跳转”的问题。request_step() 改 active_step 即可切页。
    current_step = st.radio(
        "流程步骤",
        STEP_LABELS,
        key="active_step",
        horizontal=True,
        label_visibility="collapsed",
    )
    st.caption(f"当前步骤：{current_step}　可用上方步骤条或页面底部按钮切换。")

    analysis = st.session_state.get("analysis")
    task = make_task(
        image_name,
        st.session_state.get("image_source", ""),
        st.session_state.get("task_title", ""),
        st.session_state.get("task_id", ""),
        st.session_state.get("authorization", ""),
        analysis.get("analysis_mode", analysis_mode) if analysis else analysis_mode,
    )
    # 观察卡左图必须是原始图像副本，右图才是带候选框的标注图。
    card_original = image.convert("RGB").copy()
    annotated = draw_regions(card_original, analysis.get("regions", [])) if analysis else card_original.copy()

    if current_step == STEP_LABELS[0]:
        st.markdown('<span class="section-chip">先把来源、授权和任务编号写清楚</span>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            task_id = st.text_input("任务编号", value=st.session_state.get("task_id", "BJ-2026-001"))
            title = st.text_input("任务标题", value=st.session_state.get("task_title", DEFAULT_TITLE_DEMO if use_demo else DEFAULT_TITLE_REAL))
            image_source = st.text_input("图像来源", value=st.session_state.get("image_source", ""), placeholder="例如：命题组提供，图 02")
        with c2:
            authorization = st.text_area("授权/使用说明", value=st.session_state.get("authorization", ""), height=110, placeholder="填写这张图实际适用的授权或赛事用途。")
            st.write(f"**当前图像**：{image_name}")
            st.write(f"**图像尺寸**：{image.width} × {image.height} 像素")
        st.session_state.update(task_id=task_id, task_title=title, image_source=image_source, authorization=authorization)
        st.image(image, caption="当前档案图像", width="stretch")
        st.download_button("下载原始图像副本", download_bytes, f"{Path(image_name).stem}_original.png", "image/png", width="stretch")
        st.divider()
        if st.button("下一步：AI观察 →", type="primary", width="stretch"):
            request_step(1)

    elif current_step == STEP_LABELS[1]:
        render_analysis(image, image_name, analysis_mode, deepseek_key, deepseek_model, is_synthetic=use_demo)
        if st.session_state.get("analysis"):
            st.divider()
            if st.button("下一步：人工复核 →", type="primary", width="stretch"):
                request_step(2)

    elif current_step == STEP_LABELS[2]:
        region = render_review(image, signature)
        st.divider()
        if st.button("下一步：观察卡 →", type="primary", width="stretch", disabled=region is None):
            request_step(3)

    elif current_step == STEP_LABELS[3]:
        if not analysis:
            st.info("完成AI观察和人工复核后，这里会生成单页A4观察卡。")
        else:
            review_flash = st.session_state.pop("review_flash", None)
            scroll_anchor("review-export", jump=bool(review_flash))
            if review_flash:
                st.success(review_flash)
            pdf_bytes = build_observation_card_pdf(task, card_original, annotated, analysis, st.session_state.get("audit_log", []))
            st.markdown('<span class="section-chip">一页交付，适合企业线上观察卡</span>', unsafe_allow_html=True)
            issues = card_issues(analysis, task)
            st.write("观察卡用同一编号串起整图位置、局部证据、复核决定和下一步建议。")
            if issues:
                st.warning("当前为观察草稿\n\n" + "\n".join(f"- {issue}" for issue in issues))
            st.download_button("下载观察草稿 PDF" if issues else "下载单页A4观察卡 PDF", pdf_bytes, f"{Path(image_name).stem}_observation_card.pdf", "application/pdf", type="primary", width="stretch")
            report_text = build_report(image_name, task["image_source"], analysis, task["analysis_mode"], task, st.session_state.get("audit_log", []))
            payload = {
                "task": task, "image_name": image_name, "analysis": analysis,
                "audit_log": st.session_state.get("audit_log", []),
            }
            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button("下载分析 JSON", export_json(payload), f"{Path(image_name).stem}_analysis.json", "application/json", width="stretch")
            with c2:
                st.download_button("下载 Markdown 报告", report_text, f"{Path(image_name).stem}_report.md", "text/markdown", width="stretch")
            with c3:
                st.download_button("下载候选区域图", image_bytes(annotated), f"{Path(image_name).stem}_regions.png", "image/png", width="stretch")
            if st.button("保存全部成果到项目 outputs", width="stretch"):
                output_dir = BASE_DIR / "outputs"
                save_outputs(output_dir, image_name, image, annotated, payload, report_text, pdf_bytes)
                add_audit("保存全部成果", str(output_dir))
                st.success(f"已保存到：{output_dir}")
            with st.expander("评委提问时主动说明的边界"):
                st.markdown("- AI给出的是疑似异常候选，不是文物病害最终鉴定。\n- 本项目不做虚拟修复：丢失的壁画内容不做无依据补全，只做观察、标注与人工复核。\n- 线上研发模式会调用外部平台；现场合规预览模式已禁用外部API。")
        if analysis:
            st.success("流程已完成：可下载上方观察卡、JSON 报告与区域图。")



if __name__ == "__main__":
    main()
