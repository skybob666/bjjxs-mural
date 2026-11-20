import unittest
import base64
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from ai_service import _response_detail, analyze_mural, demo_analysis, fallback_analysis, image_to_data_url, normalize_result, test_deepseek_connection
from image_utils import canvas_delta_to_bbox, create_mask, draw_regions, fabric_rectangles_to_bboxes, make_demo_mural
from network_utils import create_http_session
from local_secrets import remove_env_values, save_env_values
from observation_card import build_observation_card_pdf
from report import build_report, export_json
from review_evidence import card_issues, card_recommendations, current_review, save_region_review


class CoreSmokeTests(unittest.TestCase):
    def test_review_stays_bound_to_the_exact_saved_box(self):
        analysis = normalize_result({"regions": [{
            "bbox": [100, 200, 400, 500],
            "visible_observation": "旧框描述",
        }]})
        region = analysis["regions"][0]
        new_box = [520, 260, 670, 410]
        with self.assertRaises(ValueError):
            save_region_review(region, new_box, "已确认")
        self.assertEqual(region["bbox"], [100, 200, 400, 500])
        save_region_review(
            region, new_box, "已确认", "青绿色表面可见褐色点",
            "待比对原图细节", "记录细节", "光照不同会影响比对",
        )
        self.assertEqual(region["ai_bbox"], [100, 200, 400, 500])
        self.assertEqual(region["visible_observation"], "旧框描述")
        self.assertEqual(card_recommendations(analysis)[0]["basis"], "青绿色表面可见褐色点")
        self.assertEqual(card_issues(analysis), [])
        region["bbox"] = [530, 260, 680, 410]
        self.assertEqual(current_review(region), {})
        self.assertEqual(card_recommendations(analysis), [])
        self.assertTrue(card_issues(analysis))

    def test_two_review_outcomes_require_their_own_fields(self):
        bbox = [100, 200, 400, 500]
        region = {"bbox": bbox, "status": "待复核"}
        with self.assertRaisesRegex(ValueError, "驳回理由"):
            save_region_review(region, bbox, "已驳回")
        with self.assertRaisesRegex(ValueError, "确认或驳回"):
            save_region_review(region, bbox, "待复核")
        self.assertEqual(region["status"], "待复核")
        save_region_review(region, bbox, "已确认", "图上可见不规则浅色斑。")
        self.assertNotIn("reason", current_review(region))
        save_region_review(region, bbox, "已驳回", rejection_reason="框的位置与描述不对应。")
        self.assertEqual(current_review(region)["reason"], "框的位置与描述不对应。")
        self.assertNotIn("observation", current_review(region))

    def test_normalize_result(self):
        value = normalize_result(
            {
                "overall_observation": "ok",
                "regions": [{"bbox": [100, 200, 400, 500], "suspected_issue": "x"}],
            }
        )
        self.assertEqual(value["regions"][0]["bbox"], [100, 200, 400, 500])
        self.assertEqual(value["regions"][0]["ai_bbox"], [100, 200, 400, 500])

        rejected = normalize_result({"regions": [{"bbox": [0.1, 0.2, 0.4, 0.5]}, {"bbox": [0, 0, 1, 1]}]})
        self.assertEqual(rejected["regions"], [])

    def test_image_pipeline(self):
        image = make_demo_mural((320, 240))
        analysis = demo_analysis(image, synthetic=True)
        annotated = draw_regions(image, analysis["regions"])
        mask = create_mask(image.size, analysis["regions"][0]["bbox"])
        self.assertEqual(image.size, annotated.size)
        self.assertEqual(image.size, mask.size)
        self.assertEqual(mask.mode, "L")

    def test_exports(self):
        image = Image.new("RGB", (100, 100), "white")
        analysis = demo_analysis(image)
        report = build_report("demo.png", "synthetic", analysis)
        self.assertIn("壁鉴新生", report)
        self.assertIn("analysis", export_json({"analysis": analysis}))

    def test_canvas_and_observation_card(self):
        boxes = fabric_rectangles_to_bboxes(
            {"objects": [{"type": "rect", "left": 10, "top": 10, "width": 30, "height": 20}]},
            (100, 100),
            (200, 100),
        )
        self.assertEqual(boxes, [[100, 100, 400, 300]])
        image = make_demo_mural((320, 240))
        analysis = demo_analysis(image)
        pdf = build_observation_card_pdf(
            {"task_id": "BJ-TEST", "title": "测试观察", "image_source": "合成", "authorization": "测试", "analysis_mode": "本地"},
            image,
            draw_regions(image, analysis["regions"]),
            analysis,
            [{"action": "test", "detail": "smoke"}],
        )
        self.assertGreater(len(pdf), 1000)

    def test_canvas_delta_recovers_drawn_rectangle(self):
        background = Image.new("RGB", (100, 100), "white")
        marked = background.copy()
        ImageDraw.Draw(marked).rectangle((20, 20, 80, 60), outline="#A13B2B", width=3)
        bbox = canvas_delta_to_bbox(np.asarray(marked), background, (100, 100), (200, 100))
        self.assertIsNotNone(bbox)
        assert bbox is not None
        self.assertLessEqual(bbox[0], 210)
        self.assertLessEqual(bbox[1], 210)
        self.assertGreaterEqual(bbox[2], 790)
        self.assertGreaterEqual(bbox[3], 590)

    def test_observation_card_keeps_all_active_regions(self):
        image = make_demo_mural((1000, 800))
        analysis = demo_analysis(image, synthetic=True)
        analysis["regions"].append(
            {
                "id": 3,
                "bbox": [120, 120, 240, 260],
                "ai_bbox": None,
                "visible_observation": "人工补充的可见纹理差异。",
                "suspected_issue": "人工新增候选",
                "evidence": "人工复核时发现。",
                "uncertainty": "仍需专业人员确认。",
                "recommendation": "补拍细节照片。",
                "source": "人工框选",
                "status": "待复核",
                "human_note": "",
                "human_reviewed": False,
            }
        )
        pdf = build_observation_card_pdf(
            {"task_id": "BJ-ALL", "title": "全区域测试", "image_source": "合成", "authorization": "测试", "analysis_mode": "本地"},
            image,
            draw_regions(image, analysis["regions"]),
            analysis,
            [],
        )
        self.assertGreater(len(pdf), 1000)
        self.assertEqual(sum(1 for item in analysis["regions"] if item.get("status") != "已驳回"), 3)

    def test_deepseek_http_response(self):
        payloads = []

        class FakeResponse:
            status_code = 200
            text = ""
            headers = {}

            @staticmethod
            def json():
                return {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": (
                                    '{"overall_observation":"ok",'
                                    '"regions":[{"bbox":[100,100,300,300]}]}'
                                )
                            },
                        }
                    ]
                }

        image = Image.new("RGB", (128, 128), "white")
        class FakeSession:
            trust_env = None

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            @staticmethod
            def post(*args, **kwargs):
                payloads.append(kwargs["json"])
                return FakeResponse()

        with patch("ai_service.requests.Session", return_value=FakeSession()) as session_factory:
            result = analyze_mural(image, "test-key")
        session_factory.assert_called_once()
        self.assertEqual(result["regions"][0]["bbox"], [100, 100, 300, 300])
        self.assertEqual(payloads[0]["thinking"], {"type": "disabled"})
        self.assertEqual(payloads[0]["response_format"], {"type": "json_object"})
        self.assertEqual(payloads[0]["max_tokens"], 1200)

    def test_deepseek_connection_check(self):
        class FakeResponse:
            status_code = 200
            text = ""
            headers = {}

            @staticmethod
            def json():
                return {"data": [{"id": "deepseek-flash"}]}

        class FakeSession:
            trust_env = None

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            @staticmethod
            def get(*args, **kwargs):
                return FakeResponse()

        with patch("ai_service.requests.Session", return_value=FakeSession()):
            result = test_deepseek_connection("test-key")
        self.assertEqual(result["models"], ["deepseek-flash"])
        self.assertEqual(result["route"], "直连")

    def test_api_session_does_not_inherit_proxy_by_default(self):
        session = create_http_session()
        self.assertFalse(session.trust_env)
        session.close()

    def test_local_secret_store(self):
        with TemporaryDirectory() as folder:
            env_path = Path(folder) / ".env"
            save_env_values({"DEEPSEEK_API_KEY": "test-secret", "DEEPSEEK_MODEL": "deepseek-flash"}, env_path)
            self.assertIn("DEEPSEEK_API_KEY=test-secret", env_path.read_text(encoding="utf-8"))
            remove_env_values(["DEEPSEEK_API_KEY"], env_path)
            self.assertNotIn("DEEPSEEK_API_KEY", env_path.read_text(encoding="utf-8"))

    def test_empty_server_error_keeps_trace_id(self):
        class EmptyResponse:
            text = ""
            headers = {"x-ds-trace-id": "trace-test"}

            @staticmethod
            def json():
                return {}

        detail = _response_detail(EmptyResponse())
        self.assertIn("服务端未返回错误正文", detail)
        self.assertIn("trace-test", detail)

    def test_non_synthetic_fallback_is_explicit(self):
        image = Image.new("RGB", (2000, 1200), "white")
        result = fallback_analysis(image, "HTTP 550")
        self.assertIn("本地兜底", result["analysis_mode"])
        self.assertIn("HTTP 550", result["fallback_reason"])
        self.assertEqual(result["regions"][0]["source"], "本地兜底")
        self.assertIsNone(result["regions"][0]["ai_bbox"])

    def test_large_rgba_image_is_normalized_for_api(self):
        image = Image.new("RGBA", (4000, 3000), (220, 180, 120, 120))
        data = base64.b64decode(image_to_data_url(image).split(",", 1)[1])
        prepared = Image.open(BytesIO(data))
        self.assertEqual(prepared.mode, "RGB")
        self.assertLessEqual(max(prepared.size), 1024)


if __name__ == "__main__":
    unittest.main()
