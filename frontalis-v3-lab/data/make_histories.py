"""経験史（data/histories/*.jsonl）の生成スクリプト。

実行: python3 data/make_histories.py
決定的（固定の種）。テストはここで作ったファイルを読む。経験史は設計者が書くものなので（docs/04 第8節 4）、
どの形成史を用意したかをこのスクリプトが全部示す。結果を見てから変えた場合は CHANGELOG に書く。

人格 R / K / X は docs/04 第6節の玩具（試作の R/K/X）を、この実装の語彙に写したもの。
  R 協働に救われた : 独りで失敗 6 → 協働で成功 12 → 協働で案が採用されない 4
  K 手柄を取られた : 別々の 8 人に手柄を取られる 8 → 独りで成功 12
  X 探索者         : 探索で発見 12 → 日課で成功 8 → 他者から学ぶ 4
どの相手も、経験ごとに新しい個体（匿名の相手）として扱う。相手の数が類の有効標本数を決める。
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

OUT = Path(__file__).resolve().parent / "histories"


def exp(eid: str, ctx: str, concepts: List[str], participants: List[Dict[str, str]],
        outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"id": eid, "context": ctx, "concepts": concepts, "participants": participants,
            "outcomes": outcomes, "optional": False}


def o(t: str, actor: str, m: float) -> Dict[str, Any]:
    return {"type": t, "actor": actor, "magnitude": m}


def partner(pid: str, cls: str = "collaborator") -> List[Dict[str, str]]:
    return [{"id": pid, "class": cls}]


def write_jsonl(name: str, rows: List[Dict[str, Any]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(name: str, obj: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")


# ----------------------------------------------------------------------
# 人格 R / K / X（T2, T3, T6, T9, T1 で使う）
# ----------------------------------------------------------------------
def history_R() -> List[Dict[str, Any]]:
    rows = []
    for i in range(6):
        rows.append(exp(f"R-solo-fail-{i+1:02d}", "solo_practice", ["music", "solo_work"], [],
                        [o("autonomy_gain", "self", 1.0), o("failure", "self", 0.8)]))
    for i in range(12):
        rows.append(exp(f"R-collab-success-{i+1:02d}", "music_session", ["music", "collaboration"],
                        partner(f"r_partner{i+1:02d}"),
                        [o("connection", f"r_partner{i+1:02d}", 1.0), o("achievement", "self", 0.9)]))
    for i in range(4):
        pid = f"r_partner{13+i:02d}"
        rows.append(exp(f"R-ideas-dropped-{i+1:02d}", "project_meeting", ["collaboration"], partner(pid),
                        [o("connection", pid, 1.0), o("autonomy_loss", pid, 0.8), o("achievement", "self", 0.7)]))
    return rows


def history_K() -> List[Dict[str, Any]]:
    rows = []
    for i in range(8):
        pid = f"k_partner{i+1:02d}"
        ctx = ["music_session", "project_meeting"][i % 2]
        rows.append(exp(f"K-credit-stolen-{i+1:02d}", ctx, ["music", "collaboration", "authorship"], partner(pid),
                        [o("connection", pid, 0.6), o("credit_taken", pid, 1.0), o("betrayal", pid, 0.8),
                         o("failure", "self", 0.4)]))
    for i in range(12):
        rows.append(exp(f"K-solo-success-{i+1:02d}", "solo_practice", ["music", "solo_work", "authorship"], [],
                        [o("autonomy_gain", "self", 1.0), o("achievement", "self", 0.9)]))
    return rows


def history_X() -> List[Dict[str, Any]]:
    rows = []
    for i in range(12):
        ctx = ["travel", "workshop"][i % 2]
        rows.append(exp(f"X-discovery-{i+1:02d}", ctx, ["exploration"], [],
                        [o("autonomy_gain", "self", 1.0), o("novelty", "world", 1.0), o("achievement", "self", 0.3)]))
    for i in range(8):
        rows.append(exp(f"X-routine-success-{i+1:02d}", "solo_practice", ["routine", "music"], [],
                        [o("monotony", "world", 1.0), o("autonomy_loss", "self", 1.0), o("achievement", "self", 0.6)]))
    for i in range(4):
        pid = f"x_partner{i+1:02d}"
        rows.append(exp(f"X-learned-from-others-{i+1:02d}", "workshop", ["learning", "conversation"], partner(pid),
                        [o("connection", pid, 1.0), o("taught", pid, 0.8)]))
    return rows


def new_event() -> Dict[str, Any]:
    """3 人格に同じく与える新しい経験（SPEC 第2節の例と同じ）。"""
    return exp("new-song-with-friendA", "music_session", ["music", "collaboration"], partner("friendA"),
               [o("connection", "friendA", 1.0), o("achievement", "self", 0.8),
                o("credit_taken", "friendA", 0.8), o("taught", "friendA", 0.6)])


# ----------------------------------------------------------------------
# T4: 持続と可塑性
# ----------------------------------------------------------------------
T4_CONTEXTS = ["music_session", "project_meeting", "workshop", "studio"]


def history_T4_P1() -> List[Dict[str, Any]]:
    """8 人の共同作業者に 1 回ずつ搾取された。場面は相手と独立に巡らせる。"""
    rows = []
    for i in range(8):
        pid = f"p1_c{i+1}"
        rows.append(exp(f"T4P1-{i+1:02d}", T4_CONTEXTS[i % 4], ["collaboration", "authorship"], partner(pid),
                        [o("connection", pid, 0.4), o("credit_taken", pid, 1.0), o("betrayal", pid, 0.6)]))
    return rows


def history_T4_P2() -> List[Dict[str, Any]]:
    """普通に協働した（8 人の共同作業者と 1 回ずつ）。"""
    rows = []
    for i in range(8):
        pid = f"p2_c{i+1}"
        rows.append(exp(f"T4P2-{i+1:02d}", T4_CONTEXTS[i % 4], ["collaboration", "music"], partner(pid),
                        [o("connection", pid, 1.0), o("support", pid, 0.5), o("achievement", "self", 0.7)]))
    return rows


def history_T4_N(m: int = 32) -> List[Dict[str, Any]]:
    """新しい共同作業者 N との良い経験（m 回）。"""
    rows = []
    for i in range(m):
        rows.append(exp(f"T4N-{i+1:02d}", T4_CONTEXTS[i % 4], ["collaboration", "music"], partner("T4_N"),
                        [o("connection", "T4_N", 1.0), o("support", "T4_N", 0.7), o("achievement", "self", 0.6)]))
    return rows


# ----------------------------------------------------------------------
# T7: 同じ相手から 8 回 vs 8 人から 1 回ずつ（手柄を取る）
# ----------------------------------------------------------------------
def _t7_exp(eid: str, pid: str) -> Dict[str, Any]:
    return exp(eid, "music_session", ["music", "collaboration"], partner(pid),
               [o("connection", pid, 0.6), o("credit_taken", pid, 0.8)])


def history_T7_same() -> List[Dict[str, Any]]:
    return [_t7_exp(f"T7S-{i+1:02d}", "t7_same") for i in range(8)]


def history_T7_diff() -> List[Dict[str, Any]]:
    return [_t7_exp(f"T7D-{i+1:02d}", f"t7_diff{i+1}") for i in range(8)]


# ----------------------------------------------------------------------
# T8: 共同作業者（手柄を取る率が高い）、師、見知らぬ人（低い）を学習
# ----------------------------------------------------------------------
def history_T8_train() -> List[Dict[str, Any]]:
    rng = random.Random(8)
    contexts = ["music_session", "project_meeting", "workshop", "cafe_chat", "studio"]
    rows: List[Dict[str, Any]] = []
    for cls, n_ind in (("collaborator", 8), ("mentor", 8), ("stranger", 8)):
        for ind in range(n_ind):
            for rep in range(5):
                pid = f"t8_{cls}{ind+1}"
                k = ind * 5 + rep
                ctx = rng.choice(contexts)
                if cls == "collaborator":
                    outs = [o("connection", pid, 0.8)]
                    if k % 10 < 7:                      # 手柄を取る率 0.7
                        outs.append(o("credit_taken", pid, 0.8))
                elif cls == "mentor":
                    outs = [o("taught", pid, 0.8), o("support", pid, 0.5)]
                    if k % 20 == 0:                     # 率 0.05
                        outs.append(o("credit_taken", pid, 0.5))
                else:
                    outs = [o("novelty", pid, 0.6)]
                    if k % 20 == 7:                     # 率 0.05
                        outs.append(o("credit_taken", pid, 0.5))
                rows.append(exp(f"T8-{cls}-{ind+1}-{rep+1}", ctx, ["collaboration"] if cls != "stranger" else ["conversation"],
                                partner(pid, cls), outs))
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = f"T8-{i+1:03d}"
    return rows


def main() -> None:
    write_jsonl("R.jsonl", history_R())
    write_jsonl("K.jsonl", history_K())
    write_jsonl("X.jsonl", history_X())
    write_json("new_event.json", new_event())
    write_jsonl("T4_P1.jsonl", history_T4_P1())
    write_jsonl("T4_P2.jsonl", history_T4_P2())
    write_jsonl("T4_N.jsonl", history_T4_N(32))
    write_jsonl("T7_same.jsonl", history_T7_same())
    write_jsonl("T7_diff.jsonl", history_T7_diff())
    write_jsonl("T8_train.jsonl", history_T8_train())
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
