import unittest

from sage_app.pipelines.overlay_config import resolve_industry_overlay_config


class TestOverlayConfig(unittest.TestCase):
    def test_default_overlay_config(self):
        result = resolve_industry_overlay_config({}, trend_state=1)
        self.assertEqual(result["regime_name"], "sideways")
        self.assertAlmostEqual(result["overlay_strength"], 0.20)
        self.assertAlmostEqual(result["mainline_strength"], 0.35)
        self.assertIn("policy_score", result["signal_weights"])

    def test_regime_override_for_bull(self):
        cfg = {
            "overlay": {
                "overlay_strength": 0.2,
                "mainline_strength": 0.35,
                "signal_weights": {"policy_score": 0.4, "concept_bias": 0.3, "northbound_ratio": 0.3},
                "regime_overrides": {
                    "bull": {
                        "overlay_strength": 0.3,
                        "mainline_strength": 0.55,
                        "signal_weights": {"policy_score": 0.2, "concept_bias": 0.5, "northbound_ratio": 0.3},
                    }
                },
            }
        }
        result = resolve_industry_overlay_config(cfg, trend_state=2)
        self.assertEqual(result["regime_name"], "bull")
        self.assertAlmostEqual(result["overlay_strength"], 0.3)
        self.assertAlmostEqual(result["mainline_strength"], 0.55)
        self.assertAlmostEqual(result["signal_weights"]["concept_bias"], 0.5)

    def test_clamp_and_weight_fallback(self):
        cfg = {
            "overlay": {
                "overlay_strength": -1,
                "mainline_strength": 99,
                "signal_weights": {"policy_score": -1, "concept_bias": "bad"},
            }
        }
        result = resolve_industry_overlay_config(cfg, trend_state=0)
        self.assertEqual(result["regime_name"], "bear")
        self.assertAlmostEqual(result["overlay_strength"], 0.0)
        self.assertAlmostEqual(result["mainline_strength"], 1.0)
        self.assertIn("policy_score", result["signal_weights"])
        self.assertIn("northbound_ratio", result["signal_weights"])


if __name__ == "__main__":
    unittest.main()
