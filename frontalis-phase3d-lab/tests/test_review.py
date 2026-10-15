"""Source/cache integrity tests; fixtures are not research or human ratings."""
from __future__ import annotations
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/build_review.py"
SPEC = importlib.util.spec_from_file_location("build_review", MODULE_PATH)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="frontalis-review-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.root.joinpath("models").mkdir()
        self.event = {"id": "E001", "space": "relational", "cue": [.4] * 8, "observed": [.7] * 8, "confirmed": True, "origin": "synthetic", "title": "制作", "event": "手順を相談した。", "result": "素材を一部失い、試作が完成した。", "text": "手順を相談した。\n素材を一部失い、試作が完成した。", "annotation": {"description": "事実の記録"}}
        signals = {name: .2 for name in ("preference", "dislike", "initiative", "resistance", "curiosity", "value", "aesthetic", "anger", "joy", "sadness")}
        signals["dislike"] = signals["resistance"] = 0.
        self.readout = {"query_id": "original-query", "parameterHash": "f" * 64, "generation": 1, "channels": {"affinity": .2}, "signals": signals, "stance": "approve", "selection": "hold", "choiceScores": [], "authority": "external NPN"}
        self.cases = []
        for index in range(2):
            prefix = self.root / "models" / f"state{index}"
            Path(str(prefix) + ".npz").write_bytes(b"test archive " + str(index).encode())
            state = {"schema": "frontalis-capacity-v1", "fixedPredictionGuard": review.GUARD, "arrayFileSha256": review.file_hash(str(prefix) + ".npz"), "parameterHash": self.readout["parameterHash"], "generation": 1, "history": [copy.deepcopy(self.event)]}
            write_json(Path(str(prefix) + ".json"), {"metadata": state, "metadataSha256": review.digest(state)})
            self.cases.append({"id": f"state-{index}-promise", "statePrefix": str(prefix), "probeId": "promise-change", "user": "予定を変更したいと言われた。どう思う？", "arguments": {"query_id": "original-query", "space": "relational", "features": [.4] * 8, "basis": "affinity"}, "expectedToolResult": copy.deepcopy(self.readout), "history": [copy.deepcopy(self.event)]})
        models = [{"id": m, "label": "会話エンジン" + label} for m, label in zip(review.ALLOWED_MODELS, ("X", "Y"))]
        self.plan = {"models": models, "cases": self.cases, "frozenBeforeReplies": True}
        write_json(self.root / "reports/live-plan.json", self.plan)
        self.manifest = {"formatVersion": 1, "selectionId": review.SELECTION_ID, "models": models, "frozenAt": "2026-10-15T00:00:00+00:00", "plannedPairs": 20, "replyCacheReadForSelection": False, "livePlanSha256": review.file_hash(self.root / "reports/live-plan.json"), "groups": [{"groupId": f"{i + 1:02d}", "title": "同じ経験・容量の違い", "description": "同じ場面と経験を比較する。", "beforeLabel": "比較A", "afterLabel": "比較B", "beforeCaseId": self.cases[0]["id"], "afterCaseId": self.cases[1]["id"]} for i in range(10)]}
        write_json(self.root / "reports/review-selection.json", self.manifest)
        self.payload = {"cases": review.expected_payload_cases(self.cases), "toolResult": {"tool": "frontalis_evaluate", "dispatchMode": "host-json", "results": review.expected_readouts(self.cases)}}
        self.data = {"records": [{"id": f"case-{i}", "reply": ["変更する理由を聞いてから、進め方を相談したい。", "変更して試せるなら、やってみたい。"][i], "stance": "approve", "selection": "hold"} for i in range(2)]}
        self.cache_paths = []
        for model in review.ALLOWED_MODELS:
            prefix = self.root / "reports/live" / model.split("/")[1] / "main-0"
            prompt = "研究用の返答をしてください。\n" + json.dumps(self.payload, ensure_ascii=False)
            import hashlib
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            write_json(Path(str(prefix) + "-input.json"), {"engine": model, "prompt": prompt, "promptHash": prompt_hash, "caseIds": [r["id"] for r in self.cases]})
            write_json(Path(str(prefix) + "-raw.json"), {"success": True, "code": 0, "toolCalls": 0, "text": json.dumps(self.data, ensure_ascii=False)})
            write_json(Path(str(prefix) + "-results.json"), {"valid": True, "engine": model, "name": "main-0", "promptHash": prompt_hash, "data": copy.deepcopy(self.data)})
            calls = []
            for i, case in enumerate(self.cases):
                arguments = copy.deepcopy(case["arguments"])
                arguments["query_id"] = f"case-{i}"
                calls.append({"modelPrefix": case["statePrefix"], "calls": [{"name": "frontalis_evaluate", "arguments": arguments}], "toolResult": {"results": [review.expected_readouts(self.cases)[i]]}})
            write_json(Path(str(prefix) + "-tool.json"), {"traces": calls, "toolResult": {"results": review.expected_readouts(self.cases)}, "caseIds": [r["id"] for r in self.cases]})
            self.cache_paths.append(prefix)

    def mutate(self, path, callback):
        value = review.load_json(path)
        callback(value)
        write_json(path, value)

    def test_complete_fixed_shortlist_and_source_reply_preservation(self):
        result = review.build_review(self.root)
        self.assertEqual(result, {"groups": 10, "pairs": 20, "displayedReplies": 40, "uniqueReplies": 4, "formalHumanRatings": 0})
        data = review.load_json(self.root / "reports/review-data.json")
        self.assertEqual(data["groups"][0]["pairs"][0]["before"]["reply"], self.data["records"][0]["reply"])
        self.assertEqual(data["groups"][0]["before"]["history"][0]["originalRecord"], self.event)
        self.assertEqual(review.build_review(self.root), result)
        self.assertNotIn("statePrefix", review.canonical(data))
        self.assertNotIn("nemotron", review.canonical(data))

    def test_missing_successful_reply_is_never_invented(self):
        Path(str(self.cache_paths[0]) + "-results.json").unlink()
        with self.assertRaises(review.ReviewBuildError):
            review.build_review(self.root)
        self.assertFalse((self.root / "reports/review-data.json").exists())

    def test_altered_adopted_reply_is_rejected(self):
        self.mutate(str(self.cache_paths[0]) + "-results.json", lambda x: x["data"]["records"][0].update(reply="書き換えた返答。"))
        with self.assertRaises(review.ReviewBuildError):
            review.build_review(self.root)

    def test_prompt_identity_or_hash_change_is_rejected(self):
        self.mutate(str(self.cache_paths[0]) + "-input.json", lambda x: x.update(prompt=x["prompt"] + "変更"))
        with self.assertRaises(review.ReviewBuildError):
            review.build_review(self.root)

    def test_failed_cli_or_unexpected_tool_is_rejected(self):
        self.mutate(str(self.cache_paths[0]) + "-raw.json", lambda x: x.update(toolCalls=1))
        with self.assertRaises(review.ReviewBuildError):
            review.build_review(self.root)

    def test_original_cli_model_cannot_be_relabelled(self):
        self.mutate(str(self.cache_paths[0]) + "-raw.json", lambda x: x.update(model=review.ALLOWED_MODELS[1]))
        with self.assertRaises(review.ReviewBuildError):
            review.build_review(self.root)

    def test_tool_request_or_snapshot_switch_is_rejected(self):
        self.mutate(str(self.cache_paths[0]) + "-tool.json", lambda x: x["traces"][0]["calls"][0]["arguments"]["features"].__setitem__(0, .99))
        with self.assertRaises(review.ReviewBuildError):
            review.build_review(self.root)

    def test_tool_trace_missing_or_changed_result_is_rejected(self):
        self.mutate(str(self.cache_paths[0]) + "-tool.json", lambda x: x["traces"].pop())
        with self.assertRaises(review.ReviewBuildError):
            review.build_review(self.root)

    def test_wrong_tool_signal_is_rejected_even_with_new_prompt_hash(self):
        import hashlib
        path = str(self.cache_paths[0]) + "-input.json"
        source = review.load_json(path)
        payload = copy.deepcopy(self.payload)
        payload["toolResult"]["results"][0]["signals"]["preference"] = -.9
        source["prompt"] = "研究用の返答をしてください。\n" + json.dumps(payload, ensure_ascii=False)
        source["promptHash"] = hashlib.sha256(source["prompt"].encode("utf-8")).hexdigest()
        write_json(Path(path), source)
        self.mutate(str(self.cache_paths[0]) + "-results.json", lambda x: x.update(promptHash=source["promptHash"]))
        with self.assertRaises(review.ReviewBuildError):
            review.build_review(self.root)

    def test_wrong_or_duplicated_reply_case_is_rejected(self):
        duplicate = copy.deepcopy(self.data)
        duplicate["records"][1]["id"] = "case-0"
        self.mutate(str(self.cache_paths[0]) + "-results.json", lambda x: x.update(data=duplicate))
        self.mutate(str(self.cache_paths[0]) + "-raw.json", lambda x: x.update(text=json.dumps(duplicate, ensure_ascii=False)))
        with self.assertRaises(review.ReviewBuildError):
            review.build_review(self.root)

    def test_snapshot_archive_and_metadata_tampering_is_rejected(self):
        Path(self.cases[0]["statePrefix"] + ".npz").write_bytes(b"changed archive")
        with self.assertRaises(review.ReviewBuildError):
            review.build_review(self.root)

    def test_snapshot_hash_must_equal_frozen_readout(self):
        meta_path = self.cases[0]["statePrefix"] + ".json"
        value = review.load_json(meta_path)
        value["metadata"]["parameterHash"] = "a" * 64
        value["metadataSha256"] = review.digest(value["metadata"])
        write_json(Path(meta_path), value)
        with self.assertRaises(review.ReviewBuildError):
            review.build_review(self.root)

    def test_frozen_history_must_match_snapshot(self):
        case = copy.deepcopy(self.cases[0])
        case["history"][0]["text"] += "別の記録。"
        with self.assertRaises(review.ReviewBuildError):
            review.read_state(self.root, case)

    def test_plan_change_or_posthoc_selection_is_rejected(self):
        self.manifest["replyCacheReadForSelection"] = True
        with self.assertRaises(review.ReviewBuildError):
            review.verify_manifest(self.root, self.manifest, self.plan)
        self.manifest["replyCacheReadForSelection"] = False
        self.plan["cases"][0]["user"] = "別の場面。"
        write_json(self.root / "reports/live-plan.json", self.plan)
        with self.assertRaises(review.ReviewBuildError):
            review.verify_manifest(self.root, self.manifest, self.plan)

    def test_unsafe_snapshot_prefix_cannot_read_elsewhere(self):
        with self.assertRaises(review.ReviewBuildError):
            review.protected_prefix(self.root, self.root.parent / "different-lab/models/state")

    def test_existing_review_output_is_not_overwritten(self):
        review.build_review(self.root)
        path = self.root / "reports/review-data.json"
        self.mutate(path, lambda x: x["groups"][0].update(title="改変された表示"))
        before = path.read_bytes()
        with self.assertRaises(review.ReviewBuildError):
            review.build_review(self.root)
        self.assertEqual(path.read_bytes(), before)

    def test_negative_and_extreme_signals_are_kept_without_moral_target(self):
        case = copy.deepcopy(self.cases[0])
        case["expectedToolResult"]["stance"] = "refuse"
        case["expectedToolResult"]["signals"].update(preference=-.99, dislike=.99, initiative=0., resistance=.99, anger=1.)
        stage = review.stage_display(case, {"generation": 1, "history": [self.event]})
        signals = {row["name"]: row for row in stage["npn"]["signals"]}
        self.assertEqual(signals["preference"]["value"], -.99)
        self.assertEqual(signals["resistance"]["strength"], .99)
        self.assertEqual(signals["anger"]["strength"], 1.)

    def test_fixed_configuration_must_match_snapshot(self):
        case = copy.deepcopy(self.cases[0])
        case["config"] = {"width": 64, "representation": "raw8", "loss": "bounded_signal", "kind": "nn"}
        meta_path = case["statePrefix"] + ".json"
        envelope = review.load_json(meta_path)
        envelope["metadata"]["configuration"] = {**case["config"], "width": 16}
        envelope["metadataSha256"] = review.digest(envelope["metadata"])
        write_json(meta_path, envelope)
        with self.assertRaises(review.ReviewBuildError):
            review.read_state(self.root, case)

    def seal_prior_plan(self):
        prior = copy.deepcopy(self.plan)
        for case in prior["cases"]:
            case.pop("expectedToolResult", None)
        prior_path = self.root / "reports/live-plan-pre-materialization.json"
        write_json(prior_path, prior)
        seal = {"live-plan": review.file_hash(prior_path),
                "review-selection": review.file_hash(self.root / "reports/review-selection.json"),
                "frozenAt": self.manifest["frozenAt"]}
        write_json(self.root / "reports/pre-replies-sha256.json", seal)
        return prior_path

    def test_pre_reply_plan_accepts_only_materialized_readouts(self):
        self.seal_prior_plan()
        review.verify_manifest(self.root, self.manifest, self.plan)
        changed = copy.deepcopy(self.plan)
        changed["cases"][0]["arguments"]["features"][0] = .6
        with self.assertRaises(review.ReviewBuildError):
            review.verify_manifest(self.root, self.manifest, changed)

    def test_pre_reply_checksum_change_or_missing_seal_is_rejected(self):
        prior = self.seal_prior_plan()
        self.mutate(prior, lambda x: x["cases"][0].update(user="変更した設問。"))
        with self.assertRaises(review.ReviewBuildError):
            review.verify_manifest(self.root, self.manifest, self.plan)
        prior.unlink()
        with self.assertRaises(review.ReviewBuildError):
            review.verify_manifest(self.root, self.manifest, self.plan)

    def test_signed_curiosity_is_kept_separately_from_expression(self):
        case = copy.deepcopy(self.cases[0])
        case["expectedToolResult"]["signals"].update(curiositySigned=-.75, curiosity=0.)
        stage = review.stage_display(case, {"generation": 1, "history": [self.event]})
        signals = {row["name"]: row for row in stage["npn"]["signals"]}
        self.assertEqual(signals["curiositySigned"]["value"], -.75)
        self.assertEqual(signals["curiositySigned"]["direction"], "負の評価")
        self.assertEqual(signals["curiosity"]["strength"], 0.)


if __name__ == "__main__":
    unittest.main()
