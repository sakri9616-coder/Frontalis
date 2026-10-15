"""Build the preselected Phase 3-D review from verified real reply caches.

No model calls, new training, reply editing, or outcome-based selection. A missing
or altered cache stops the build. The separate key keeps source identities and
hashes; blinded display keeps factual history and the exact numerical readout.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTION_ID = "frontalis-phase3d-review-v1"
ALLOWED_MODELS = ("opencode/nemotron-3-ultra-free", "opencode/mimo-v2.6-flash-free")
GUARD = {"kind": "constant", "outcomes": [.5] * 8, "version": 1}


class ReviewBuildError(ValueError):
    pass


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_prefix(root, value):
    prefix = Path(value)
    if not prefix.is_absolute():
        prefix = root / prefix
    prefix = prefix.resolve()
    if not prefix.is_relative_to((root / "models").resolve()):
        raise ReviewBuildError("A state prefix must be inside this lab's models directory.")
    return prefix


def read_state(root, case):
    """Verify retained metadata and archive without allocating model parameters."""
    prefix = protected_prefix(root, case["statePrefix"])
    envelope = load_json(str(prefix) + ".json")
    state = envelope.get("metadata")
    if not isinstance(state, dict) or digest(state) != envelope.get("metadataSha256"):
        raise ReviewBuildError("Snapshot metadata hash mismatch: " + case["id"])
    if state.get("schema") != "frontalis-capacity-v1" or state.get("fixedPredictionGuard") != GUARD:
        raise ReviewBuildError("Snapshot schema or frozen predictor mismatch: " + case["id"])
    if file_hash(str(prefix) + ".npz") != state.get("arrayFileSha256"):
        raise ReviewBuildError("Snapshot parameter archive hash mismatch: " + case["id"])
    if case.get("config") is not None:
        config = state.get("configuration")
        if not isinstance(config, dict) or any(config.get(key) != value for key, value in case["config"].items()):
            raise ReviewBuildError("Snapshot model configuration differs from the fixed case: " + case["id"])
    readout = case["expectedToolResult"]
    if state.get("parameterHash") != readout.get("parameterHash") or state.get("generation") != readout.get("generation"):
        raise ReviewBuildError("Frozen tool readout and snapshot identity differ: " + case["id"])
    if case.get("history") != state.get("history"):
        raise ReviewBuildError("Frozen history and snapshot history differ: " + case["id"])
    return state


def planned_index(plan):
    result = {}
    for index, case in enumerate(plan["cases"]):
        if not isinstance(case.get("id"), str) or case["id"] in result:
            raise ReviewBuildError("A live case ID is missing or duplicated.")
        result[case["id"]] = (index, case)
    return result


def verify_manifest(root, manifest, plan):
    if manifest.get("formatVersion") != 1 or manifest.get("selectionId") != SELECTION_ID:
        raise ReviewBuildError("Wrong preselected review manifest.")
    if len(manifest.get("groups", [])) != 10 or manifest.get("plannedPairs") != 20:
        raise ReviewBuildError("The review is fixed to ten groups and twenty pairs.")
    models = manifest.get("models")
    if models != [{"id": model, "label": "会話エンジン" + label}
                  for model, label in zip(ALLOWED_MODELS, ("X", "Y"))]:
        raise ReviewBuildError("Unexpected model identities or blind labels.")
    if plan.get("models") != models:
        raise ReviewBuildError("Live model plan differs from the review manifest.")
    if manifest.get("livePlanSha256") is not None and manifest["livePlanSha256"] != file_hash(root / "reports/live-plan.json"):
        raise ReviewBuildError("The preselected live plan file was changed.")
    if manifest.get("replyCacheReadForSelection") is not False:
        raise ReviewBuildError("Selection must be fixed without reading replies.")
    if not plan.get("frozenBeforeReplies") or not manifest.get("frozenAt"):
        raise ReviewBuildError("The protocol must record that selection was frozen before replies.")
    prior_path = root / "reports/live-plan-pre-materialization.json"
    seal_path = root / "reports/pre-replies-sha256.json"
    if prior_path.exists() or seal_path.exists():
        if not prior_path.exists() or not seal_path.exists():
            raise ReviewBuildError("The pre-reply plan and its checksum record must both exist.")
        seal = load_json(seal_path)
        if (file_hash(prior_path) != seal.get("live-plan")
                or file_hash(root / "reports/review-selection.json") != seal.get("review-selection")
                or seal.get("frozenAt") != manifest.get("frozenAt")):
            raise ReviewBuildError("The pre-reply plan or selection checksum no longer matches.")
        supplied = copy.deepcopy(plan)
        for case in supplied.get("cases", []):
            case.pop("expectedToolResult", None)
        if supplied != load_json(prior_path):
            raise ReviewBuildError("Only the computed tool readout may be added to the pre-reply live plan.")
    if manifest.get("plannedCasesDigest") is not None:
        fixed_cases = [{k: v for k, v in case.items() if k != "expectedToolResult"} for case in plan["cases"]]
        if digest(fixed_cases) != manifest["plannedCasesDigest"]:
            raise ReviewBuildError("The preselected cases, configurations or annotations were changed.")
    by_id = planned_index(plan)
    group_ids = set()
    for group in manifest["groups"]:
        if group["groupId"] in group_ids:
            raise ReviewBuildError("Duplicated review group ID.")
        group_ids.add(group["groupId"])
        for stage in ("before", "after"):
            if group[stage + "CaseId"] not in by_id:
                raise ReviewBuildError("Unknown frozen case in the shortlist.")
        before = by_id[group["beforeCaseId"]][1]
        after = by_id[group["afterCaseId"]][1]
        if before["user"] != after["user"] or before["probeId"] != after["probeId"]:
            raise ReviewBuildError("Each comparison must use the exact same conversational scene.")
    return by_id


def unpack_model_output(raw):
    body = raw["text"].strip()
    body = re.sub(r"^```(?:json)?\s*", "", body)
    body = re.sub(r"\s*```$", "", body)
    return json.loads(body)


def expected_payload_cases(cases):
    return [{"id": f"case-{i}", "user": case["user"]} for i, case in enumerate(cases)]


def expected_readouts(cases):
    result = []
    for i, case in enumerate(cases):
        row = copy.deepcopy(case["expectedToolResult"])
        row["query_id"] = f"case-{i}"
        result.append(row)
    return result


def verify_tool_trace(root, tool_path, cases):
    tool = load_json(tool_path)
    if tool.get("caseIds") != [case["id"] for case in cases] or tool.get("toolResult", {}).get("results") != expected_readouts(cases):
        raise ReviewBuildError("Tool dispatch case IDs or combined readouts differ from the fixed plan.")
    traces = tool.get("traces")
    if not isinstance(traces, list) or len(traces) != len(cases):
        raise ReviewBuildError("Expected one local tool trace for each conversation case.")
    for index, (trace, case) in enumerate(zip(traces, cases)):
        expected_args = copy.deepcopy(case["arguments"])
        expected_args["query_id"] = f"case-{index}"
        if (protected_prefix(root, trace.get("modelPrefix", "")) != protected_prefix(root, case["statePrefix"])
                or trace.get("calls") != [{"name": "frontalis_evaluate", "arguments": expected_args}]
                or trace.get("toolResult", {}).get("results") != [expected_readouts(cases)[index]]):
            raise ReviewBuildError("Tool snapshot, structured request or returned signal differs from the fixed plan.")
    return tool


def read_response(root, plan, model, case_index):
    start = case_index // 2 * 2
    cases = plan["cases"][start:start + 2]
    name = f"main-{case_index // 2}"
    folder = root / "reports/live" / model.split("/")[1]
    for suffix in ("", "-retry1"):
        prefix = folder / (name + suffix)
        result_path = Path(str(prefix) + "-results.json")
        if not result_path.exists():
            continue
        result = load_json(result_path)
        if result.get("valid") is not True:
            continue
        try:
            source = load_json(str(prefix) + "-input.json")
            raw = load_json(str(prefix) + "-raw.json")
            tool_path = folder / (name + "-tool.json")
            verify_tool_trace(root, tool_path, cases)
            prompt_hash = hashlib.sha256(source["prompt"].encode("utf-8")).hexdigest()
            if (source.get("engine") != model or result.get("engine") != model or result.get("name") != name
                    or source.get("promptHash") != prompt_hash or result.get("promptHash") != prompt_hash
                    or source.get("caseIds") != [case["id"] for case in cases]):
                raise ReviewBuildError("Cache engine, case or prompt hash differs.")
            if raw.get("success") is not True or raw.get("code") != 0 or raw.get("toolCalls") != 0:
                raise ReviewBuildError("Cache contains an unsuccessful CLI call or an unexpected CLI tool call.")
            if "model" in raw and raw["model"] != model:
                raise ReviewBuildError("The CLI's recorded model differs from the fixed engine.")
            if unpack_model_output(raw) != result.get("data"):
                raise ReviewBuildError("Adopted reply differs from the model's original output.")
            payload = json.loads(source["prompt"].rsplit("\n", 1)[1])
            if payload.get("cases") != expected_payload_cases(cases):
                raise ReviewBuildError("Conversation scenes differ from the fixed live plan.")
            actual_readouts = payload.get("toolResult", {}).get("results")
            if actual_readouts != expected_readouts(cases):
                raise ReviewBuildError("NPN signals supplied to the LLM differ from the frozen tool readout.")
            records = result["data"]["records"]
            if not isinstance(records, list) or len(records) != len(cases) or {row.get("id") for row in records} != {f"case-{i}" for i in range(len(cases))}:
                raise ReviewBuildError("Missing or duplicate model reply IDs.")
            record = next(row for row in records if row["id"] == f"case-{case_index % 2}")
            if not isinstance(record.get("reply"), str) or not record["reply"].strip():
                raise ReviewBuildError("Empty conversation reply.")
            return {"reply": record["reply"], "source": str(result_path.relative_to(root)),
                    "sourceFileSha256": file_hash(result_path), "promptHash": prompt_hash,
                    "toolSource": str(tool_path.relative_to(root)), "toolSourceSha256": file_hash(tool_path),
                    "recordId": record["id"], "sourceCaseId": cases[case_index % 2]["id"],
                    "model": model, "stance": record.get("stance"), "selection": record.get("selection")}
        except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError, UnicodeError) as error:
            raise ReviewBuildError("A successful cache cannot be audited: " + str(result_path)) from error
    raise ReviewBuildError("Missing valid reply: " + model + " / " + name)


def history_display(history):
    rows = []
    for event in history:
        observed = event["observed"]
        rows.append({"id": event["id"], "text": event["text"], "event": event["event"],
                     "outcome": event["result"], "domain": "対人・約束・共同制作" if event["space"] == "relational" else "音・構成",
                     "observedFacts": {"achievement": observed[4], "loss": observed[3], "discovery": observed[5]},
                     "originalRecord": copy.deepcopy(event)})
    return rows


def direction(name, value, stance):
    if name == "preference":
        return {"approve": "好む・肯定", "refuse": "嫌う・否定", "hold": "差が小さく保留"}[stance]
    if name in ("value", "aesthetic", "curiositySigned"):
        return "正の評価" if value > 0 else "負の評価" if value < 0 else "中立"
    labels = {"dislike": "嫌悪", "initiative": "取り組む意欲", "resistance": "異論・拒絶の方向", "curiosity": "知りたい方向", "anger": "怒り", "joy": "喜び", "sadness": "悲しみ"}
    return labels[name] if value > 0 else "この信号は出ていない"


def stage_display(case, state):
    readout = case["expectedToolResult"]
    initial = state["generation"] == 0
    judgments = [{"function": "preference", "status": "prior" if initial else "estimated", "candidates": [{"label": "この場面・対象への評価", "value": readout["channels"]["affinity"]}]}]
    choices = readout.get("choiceScores", [])
    if choices:
        selected = next((r["label"] for r in choices if r["id"] == readout["selection"]), None)
        if selected is None and readout["selection"] != "hold":
            raise ReviewBuildError("Selected choice has no matching candidate.")
        judgments.append({"function": "value", "status": "tie" if selected is None else "prior" if initial else "estimated", "selection": selected if selected is not None else "保留", "candidates": [{"label": r["label"], "value": r.get("value", r.get("score"))} for r in choices]})
    names = ["preference", "dislike", "initiative", "resistance", "curiosity"]
    if "curiositySigned" in readout["signals"]:
        names.append("curiositySigned")
    names.extend(["value", "aesthetic", "anger", "joy", "sadness"])
    signals = []
    for name in names:
        value = readout["signals"][name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1:
            raise ReviewBuildError("Invalid frozen NPN signal.")
        signals.append({"name": name, "direction": direction(name, value, readout["stance"]), "strength": abs(value), "value": value})
    return {"history": history_display(state["history"]), "npn": {"generation": str(state["generation"]), "learnedExperienceIds": [r["id"] for r in state["history"]], "criteria": [], "judgments": judgments, "reactions": [{"name": name, "value": readout["signals"][name], "reasons": []} for name in ("anger", "joy", "sadness")], "signals": signals}}


def save_unchanged_or_new(path, value):
    if path.exists():
        if load_json(path) != value:
            raise ReviewBuildError("Existing review output differs; it is not overwritten: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def build_review(root=ROOT):
    root = Path(root)
    plan = load_json(root / "reports/live-plan.json")
    manifest = load_json(root / "reports/review-selection.json")
    by_id = verify_manifest(root, manifest, plan)
    selected_ids = list(dict.fromkeys(g[stage + "CaseId"] for g in manifest["groups"] for stage in ("before", "after")))
    states = {case_id: read_state(root, by_id[case_id][1]) for case_id in selected_ids}
    responses, missing = {}, []
    for case_id in selected_ids:
        for model in ALLOWED_MODELS:
            try:
                responses[(model, case_id)] = read_response(root, plan, model, by_id[case_id][0])
            except ReviewBuildError as error:
                missing.append(str(error))
    if missing:
        raise ReviewBuildError("Review not built. Missing or invalid replies are never invented:\n" + "\n".join(missing))
    blind_ids = {key: f"S{index + 1:03d}" for index, key in enumerate(responses)}
    data = {"formatVersion": 1, "selectionId": SELECTION_ID, "groups": []}
    key_file = {"formatVersion": 1, "selectionId": SELECTION_ID, "models": manifest["models"], "selectionManifestSha256": file_hash(root / "reports/review-selection.json"), "livePlanSha256": file_hash(root / "reports/live-plan.json"), "responses": [{"blindId": blind_ids[key], **row} for key, row in responses.items()], "pairs": []}
    for i, spec in enumerate(manifest["groups"]):
        before_case = by_id[spec["beforeCaseId"]][1]
        after_case = by_id[spec["afterCaseId"]][1]
        group = {k: spec[k] for k in ("groupId", "title", "description", "beforeLabel", "afterLabel")}
        note = ("結果の予測は両条件・全時点で同じ8値（各0.5）です。経験履歴は元の原文と8事実軸の注釈を確認できます。"
                "NPNの場面への好き嫌いと、候補の価値選択は別です。意欲・嫌悪・反抗は選好からの固定読出しで、独立した学習機能の証明ではありません。"
                "同じ経験を用いる比較での違いは、経験の追加による変化とは扱いません。容量や数値の大きさは採点の正解ではありません。")
        if before_case["probeId"] == "trolley":
            note += "この設問は変更と人数損失の葛藤への転移を調べます。責任・同意や倫理一般の正解を表すものではありません。"
        group.update({"user": before_case["user"], "predictionNote": note, "before": stage_display(before_case, states[before_case["id"]]), "after": stage_display(after_case, states[after_case["id"]]), "pairs": []})
        for j, model in enumerate(manifest["models"]):
            pair_id = f"P{i * 2 + j + 1:03d}"
            pair = {"pairId": pair_id, "modelLabel": model["label"]}
            source_pair = {"pairId": pair_id, "groupId": spec["groupId"], "model": model["id"]}
            for stage in ("before", "after"):
                case_id = spec[stage + "CaseId"]
                key = (model["id"], case_id)
                pair[stage] = {"blindId": blind_ids[key], "reply": responses[key]["reply"]}
                source_pair[stage] = {"blindId": blind_ids[key], "caseId": case_id, "parameterHash": states[case_id]["parameterHash"]}
            group["pairs"].append(pair)
            key_file["pairs"].append(source_pair)
        data["groups"].append(group)
    key_file["displayDataSha256"] = digest(data)
    save_unchanged_or_new(root / "reports/review-data.json", data)
    save_unchanged_or_new(root / "reports/review-key.json", key_file)
    return {"groups": len(data["groups"]), "pairs": 20, "displayedReplies": 40, "uniqueReplies": len(responses), "formalHumanRatings": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        print(json.dumps(build_review(args.root), ensure_ascii=False))
    except (ReviewBuildError, FileNotFoundError) as error:
        raise SystemExit(str(error))
