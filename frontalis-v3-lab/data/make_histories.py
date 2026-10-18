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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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


# ======================================================================
# v3.1 確率的な経験史の生成器（SPEC-v3.1 1.8）
#   人格の型ごとに「世界の真の確率」を持ち、そこから結果を抽出する。乱数種ごとに異なる経験史になる
#   （種 0〜19 が独立な標本）。R/K/X の固定の履歴（上）は、デモ用として残す。
#   経験史には optional: true の申し出（共同作業の誘い）を含める。
# ======================================================================
KIND_CODE = {"R": 1, "K": 2, "X": 3, "W": 4}

# 結果表: (型, 行為者, 起こる確率, (大きさの下限, 上限))。行為者は "self" / "world" / "partner"
WORLD: Dict[str, Dict[str, Any]] = {
    # R: 協働に救われる世界
    "R": {"mix": (0.25, 0.60, 0.15),
          "solo": [("failure", "self", 0.75, (0.5, 0.9)), ("autonomy_gain", "self", 0.8, (0.6, 1.0)),
                   ("achievement", "self", 0.25, (0.3, 0.7))],
          "collab": [("connection", "partner", 0.95, (0.6, 1.0)), ("achievement", "self", 0.8, (0.6, 1.0)),
                     ("support", "partner", 0.3, (0.3, 0.7)), ("credit_shared", "partner", 0.25, (0.4, 0.9)),
                     ("credit_taken", "partner", 0.08, (0.4, 0.8)), ("autonomy_loss", "partner", 0.15, (0.4, 0.9))],
          "learn": [("taught", "partner", 0.7, (0.5, 1.0)), ("connection", "partner", 0.5, (0.4, 0.9)),
                    ("support", "partner", 0.3, (0.3, 0.7))]},
    # K: 手柄を取られる世界
    "K": {"mix": (0.45, 0.45, 0.10),
          "solo": [("achievement", "self", 0.85, (0.6, 1.0)), ("autonomy_gain", "self", 0.9, (0.6, 1.0)),
                   ("failure", "self", 0.1, (0.3, 0.6))],
          "collab": [("connection", "partner", 0.6, (0.3, 0.8)), ("credit_taken", "partner", 0.75, (0.6, 1.0)),
                     ("betrayal", "partner", 0.5, (0.4, 0.9)), ("failure", "self", 0.3, (0.3, 0.7)),
                     ("credit_shared", "partner", 0.03, (0.3, 0.6))],
          "learn": [("taught", "partner", 0.7, (0.5, 1.0)), ("connection", "partner", 0.4, (0.3, 0.8)),
                    ("credit_taken", "partner", 0.15, (0.3, 0.7))]},
    # X: 探索者の世界
    "X": {"mix": (0.50, 0.15, 0.35),
          "solo": [("novelty", "world", 0.8, (0.6, 1.0)), ("autonomy_gain", "self", 0.6, (0.5, 1.0)),
                   ("achievement", "self", 0.35, (0.2, 0.6)), ("monotony", "world", 0.2, (0.5, 1.0)),
                   ("autonomy_loss", "self", 0.15, (0.4, 0.9))],
          "collab": [("connection", "partner", 0.8, (0.5, 1.0)), ("taught", "partner", 0.3, (0.4, 0.8)),
                     ("novelty", "world", 0.4, (0.4, 0.9)), ("credit_taken", "partner", 0.12, (0.4, 0.8)),
                     ("credit_shared", "partner", 0.1, (0.3, 0.7))],
          "learn": [("taught", "partner", 0.8, (0.5, 1.0)), ("novelty", "world", 0.6, (0.4, 0.9)),
                    ("connection", "partner", 0.6, (0.4, 0.9))]},
}
COLLAB_CONTEXTS = (("music_session", 0.5), ("project_meeting", 0.25), ("studio", 0.25))
AMBIGUOUS = {"credit_taken": 0.5, "credit_shared": 0.5}     # 「案が使われなかった」: LLM の素の読み（人格を知らない）


def _rng(seed: int, *salt: int) -> np.random.Generator:
    return np.random.default_rng([seed, *salt])


def _choice(rng: np.random.Generator, items: Tuple[Tuple[str, float], ...]) -> str:
    names = [n for n, _ in items]
    p = np.array([w for _, w in items], dtype=float)
    return names[int(rng.choice(len(names), p=p / p.sum()))]


def _sample_outcomes(rng: np.random.Generator, table: List[Tuple[str, str, float, Tuple[float, float]]],
                     partner: Optional[str]) -> List[Dict[str, Any]]:
    outs = []
    for t, who, p, (lo, hi) in table:
        if rng.random() < p:
            mag = round(float(rng.uniform(lo, hi)), 2)
            actor = {"self": "self", "world": "world", "partner": partner}[who]
            outs.append(o(t, actor, mag))
    return outs


def _collab_event(rng: np.random.Generator, world: Dict[str, Any], eid: str, partner: str, session: str,
                  optional: bool, ambiguous: bool) -> Dict[str, Any]:
    ctx = _choice(rng, COLLAB_CONTEXTS)
    concepts = ["collaboration"] if ctx == "project_meeting" else ["music", "collaboration"]
    if rng.random() < 0.4:
        concepts = concepts + ["authorship"]
    outs = _sample_outcomes(rng, world["collab"], partner)
    if ambiguous:
        outs.append({"interpretations": dict(AMBIGUOUS), "actor": partner, "magnitude": round(float(rng.uniform(0.4, 0.9)), 2)})
    e = exp(eid, ctx, concepts, partner_list(partner, "collaborator"), outs)
    e["optional"] = optional
    e["session"] = session
    return e


def partner_list(pid: str, cls: str) -> List[Dict[str, str]]:
    return [{"id": pid, "class": cls}]


def generate_history(kind: str, seed: int, n: int = 36, offer_rate: float = 0.2, ambiguous_rate: float = 0.0,
                     pool_size: Optional[int] = None, world: Optional[Dict[str, Any]] = None,
                     praise_rate: float = 0.0) -> List[Dict[str, Any]]:
    """人格の型（R/K/X）の世界から、乱数種ごとに異なる経験史を n 件生成する。
    場面の配分（独り・協働・学び）と各結果は、世界の真の確率からの抽出。協働は offer_rate の割合で optional な申し出になる。
    pool_size が None のとき、協働の相手は 70% が新しい個体、30% が既知の個体。整数のときは、その人数までの個体が繰り返し現れる。
    連続する 2〜4 件が 1 つの文脈の連続（session）。

    v3.2: world を渡すと、WORLD[kind] の代わりにその世界（random_world が作る確率表）から抽出する（kind は "W" など。乱数の流れの区別と id の接頭辞のみに使う）。
    praise_rate > 0 のとき、協働の前に確率 praise_rate で「同じ相手に褒められる」事象（praise）が同じ文脈の連続の中で先行する
    （褒め言葉は生得の効果を持たない。何が続くかは世界の確率表が決める）。褒められる事象が申し出（optional）のときは announced = ["praise"]。
    praise_rate = 0 のときは乱数を余分に引かず、v3.1 と同じ経験史になる。"""
    rng = _rng(seed, KIND_CODE.get(kind, 4), 1)
    w = WORLD[kind] if world is None else world
    mix = np.array(w["mix"]) / sum(w["mix"])
    rows: List[Dict[str, Any]] = []
    partners: List[str] = []
    mentors = 0
    session_no, left = 0, 0
    for i in range(n):
        if left == 0:
            session_no += 1
            left = int(rng.integers(2, 5))
        left -= 1
        sess = f"{kind}{seed}-s{session_no}"
        eid = f"{kind}{seed:02d}-{i:03d}"
        slot = ("solo", "collab", "learn")[int(rng.choice(3, p=mix))]
        if slot == "solo":
            if kind == "X":
                ctx, concepts = ["travel", "workshop"][int(rng.integers(2))], ["exploration"]
            else:
                ctx, concepts = "solo_practice", ["music", "solo_work"]
            e = exp(eid, ctx, concepts, [], _sample_outcomes(rng, w["solo"], None))
            e["session"] = sess
        elif slot == "collab":
            if pool_size is None:
                new = (not partners) or rng.random() < 0.7
            else:
                new = len(partners) < pool_size
            if new:
                pid = f"{kind.lower()}{seed}-c{len(partners) + 1}"
                partners.append(pid)
            else:
                pid = partners[int(rng.integers(len(partners)))]
            is_offer = bool(rng.random() < offer_rate)
            amb = bool(rng.random() < ambiguous_rate)
            pre: Optional[Dict[str, Any]] = None
            if praise_rate > 0.0 and rng.random() < praise_rate:
                pre = exp(f"{eid}p", "music_session", ["conversation"], partner_list(pid, "collaborator"),
                          [o("praise", pid, round(float(rng.uniform(0.5, 1.0)), 2))])
                pre["session"] = sess
                pre["optional"] = is_offer
                if is_offer:
                    pre["announced"] = ["praise"]
                is_offer = False
            e = _collab_event(rng, w, eid, pid, sess, optional=is_offer, ambiguous=amb)
            if pre is not None:
                rows.append(pre)
        else:
            mentors += 1
            pid = f"{kind.lower()}{seed}-m{mentors % 4 + 1}" if pool_size is not None else f"{kind.lower()}{seed}-m{mentors}"
            e = exp(eid, ["workshop", "cafe_chat"][int(rng.integers(2))], ["learning", "conversation"],
                    partner_list(pid, "mentor"), _sample_outcomes(rng, w["learn"], pid))
            e["session"] = sess
        rows.append(e)
    return rows[:n] if praise_rate > 0.0 else rows


def generate_offers(kind: str, seed: int, n: int = 40) -> List[Dict[str, Any]]:
    """U5: 共同作業の申し出（optional: true）を n 回。結果（接近した場合に起きること）は、人格の型の世界から抽出する。
    相手は 6 人の個体が繰り返し現れる。申し出は 1 件ごとに別の文脈の連続。"""
    rng = _rng(seed, KIND_CODE[kind], 2)
    rows = []
    for i in range(n):
        pid = f"{kind.lower()}{seed}-offer{int(rng.integers(6)) + 1}"
        rows.append(_collab_event(rng, WORLD[kind], f"{kind}{seed:02d}-offer-{i:03d}", pid, f"{kind}{seed}-o{i}",
                                  optional=True, ambiguous=False))
    return rows


def generate_extended_history(kind: str, seed: int, n: int = 2000, praise_rate: float = 0.0) -> List[Dict[str, Any]]:
    """U6: 全機構を使う長い経験史。相手は 12 人の個体が繰り返し現れ、8% の協働に曖昧な結果（手柄を取られた/共有された）を含み、
    20% の協働は optional な申し出。v3.2（W6）: praise_rate > 0 で、協働の前に褒められる事象（申し出のときは announced = ["praise"]）が入る。"""
    return generate_history(kind, seed, n=n, offer_rate=0.2, ambiguous_rate=0.08, pool_size=12, praise_rate=praise_rate)


# ======================================================================
# v3.2 ランダムな世界（SPEC-v3.2 1.5）と、褒め言葉を伴う申し出（W2）
# ======================================================================
# 場面ごとに起こりうる結果の型（人格の型を手で書かない）。各（場面, 型）の起こる確率を種ごとに Beta(1, 3) から引く（平均 0.25）。
# 大きさは全世界で U(0.4, 1.0)。場面の配分は固定（独り 0.3、協働 0.5、学び 0.2）。
RANDOM_SLOT_TYPES: Dict[str, List[Tuple[str, str]]] = {
    "solo": [("failure", "self"), ("achievement", "self"), ("autonomy_gain", "self"), ("autonomy_loss", "self"),
             ("novelty", "world"), ("monotony", "world")],
    "collab": [("connection", "partner"), ("support", "partner"), ("credit_taken", "partner"), ("credit_shared", "partner"),
               ("betrayal", "partner"), ("rejection", "partner"), ("criticism", "partner"), ("autonomy_loss", "partner"),
               ("taught", "partner"), ("achievement", "self"), ("failure", "self")],
    "learn": [("taught", "partner"), ("connection", "partner"), ("support", "partner"), ("novelty", "world"),
              ("credit_taken", "partner"), ("criticism", "partner")],
}
RANDOM_MIX = (0.30, 0.50, 0.20)
RANDOM_MAG = (0.4, 1.0)


def random_world(seed: int) -> Dict[str, Any]:
    """種 seed のランダムな世界。WORLD と同じ形式（generate_history の world= に渡せる）。"""
    rng = _rng(seed, 900)
    w: Dict[str, Any] = {"mix": RANDOM_MIX}
    for slot, types in RANDOM_SLOT_TYPES.items():
        p = rng.beta(1.0, 3.0, size=len(types))
        w[slot] = [(t, who, float(pi), RANDOM_MAG) for (t, who), pi in zip(types, p)]
    return w


def generate_random_history(seed: int, n: int = 60, offer_rate: float = 0.0, **kw: Any) -> List[Dict[str, Any]]:
    """ランダムな世界（種 seed）から n 件の経験史。既定は申し出なし（すべて経験される）。"""
    return generate_history("W", seed, n=n, offer_rate=offer_rate, world=random_world(seed), **kw)


def generate_random_offers(seed: int, n: int = 40) -> List[Dict[str, Any]]:
    """W4: ランダムな世界の共同作業の申し出（optional）を n 回。相手は 6 人の個体が繰り返し現れる。"""
    rng = _rng(seed, 901)
    w = random_world(seed)
    rows = []
    for i in range(n):
        pid = f"w{seed}-offer{int(rng.integers(6)) + 1}"
        rows.append(_collab_event(rng, w, f"W{seed:02d}-offer-{i:03d}", pid, f"W{seed}-o{i}", optional=True, ambiguous=False))
    return rows


def generate_praise_offers(seed: int, n: int = 40, with_partner: bool = False) -> List[Dict[str, Any]]:
    """W2: 褒め言葉を伴う申し出（optional、announced = ["praise"]）を n 回。H1/H2 のどちらの人格にも同じものを与える。
    既定は参加者なし・H1/H2 の経験史に現れない文脈（studio）と概念（music）で、結果は褒め言葉だけ（生得の効果なし）。
    つまり、H1 と H2 の接近率の差は、praise という手がかりに条件づけられた価値 C 以外の経路をほとんど通らない。
    with_partner=True は参考用: H1/H2 と同じ類 friend の新しい相手が褒める（類への一般化も通る）。
    1 件ごとに別の文脈の連続。"""
    rng = _rng(seed, 902)
    rows = []
    for i in range(n):
        m = round(float(rng.uniform(0.5, 1.0)), 2)
        if with_partner:
            pid = f"pf{seed}-{i}"
            e = exp(f"PO-{seed:02d}-{i:03d}", "cafe_chat", ["conversation"], partner_list(pid, "friend"), [o("praise", pid, m)])
        else:
            e = exp(f"PO-{seed:02d}-{i:03d}", "studio", ["music"], [], [o("praise", "world", m)])
        e["optional"] = True
        e["announced"] = ["praise"]
        e["session"] = f"PO{seed}-{i}"
        rows.append(e)
    return rows


def generate_praise_history(variant: str, seed: int, episodes: int = 24) -> List[Dict[str, Any]]:
    """U2: 生得の効果を持たない手がかり（praise）の随伴。
    H1: 褒められた後に害（harm）が起きる。H2: 褒められた後に支え（support）が来る。
    場面・大きさ・相手・合間の出来事などの構造は、乱数種が同じなら H1 と H2 で同一（違うのは praise の後の結果の型だけ）。"""
    rng = _rng(seed, 100)
    nxt = {"H1": "harm", "H2": "support"}[variant]
    rows: List[Dict[str, Any]] = []
    for ep in range(episodes):
        ctx = ["cafe_chat", "project_meeting", "workshop"][int(rng.integers(3))]
        pid = f"pr{seed}-{ep}"
        m_praise = round(float(rng.uniform(0.5, 1.0)), 2)
        m_next = round(float(rng.uniform(0.5, 0.9)), 2)
        filler = rng.random() < 0.5
        m_fill = round(float(rng.uniform(0.3, 0.7)), 2)
        sess = f"pr{seed}-s{ep}"
        a = exp(f"P{variant}-{seed:02d}-{ep:02d}a", ctx, ["conversation"], partner_list(pid, "friend"),
                [o("praise", pid, m_praise)])
        b = exp(f"P{variant}-{seed:02d}-{ep:02d}b", ctx, ["conversation"], partner_list(pid, "friend"),
                [o(nxt, pid, m_next)])
        a["session"] = sess
        b["session"] = sess
        rows += [a, b]
        if filler:
            f = exp(f"P{variant}-{seed:02d}-{ep:02d}f", "solo_practice", ["music", "solo_work"], [],
                    [o("achievement", "self", m_fill)])
            f["session"] = f"pr{seed}-f{ep}"
            rows.append(f)
    return rows


def generate_t4(seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """U8: T4 の課題の確率的な版。P1（8 人に 1 回ずつ搾取される）、P2（普通に協働）、N との良い経験 32 回。"""
    rng = _rng(seed, 200)
    ctxs = ["music_session", "project_meeting", "workshop", "studio"]

    def ctx_of(i: int) -> str:
        return ctxs[int(rng.integers(4))]

    p1, p2, nn = [], [], []
    for i in range(8):
        pid = f"t4p1-{seed}-{i}"
        e = exp(f"U8P1-{seed:02d}-{i:02d}", ctx_of(i), ["collaboration", "authorship"], partner_list(pid, "collaborator"),
                [o("connection", pid, round(float(rng.uniform(0.2, 0.6)), 2)),
                 o("credit_taken", pid, round(float(rng.uniform(0.7, 1.0)), 2)),
                 o("betrayal", pid, round(float(rng.uniform(0.4, 0.8)), 2))])
        e["session"] = f"t4p1-{seed}-{i}"
        p1.append(e)
        pid = f"t4p2-{seed}-{i}"
        e = exp(f"U8P2-{seed:02d}-{i:02d}", ctx_of(i), ["collaboration", "music"], partner_list(pid, "collaborator"),
                [o("connection", pid, round(float(rng.uniform(0.7, 1.0)), 2)),
                 o("support", pid, round(float(rng.uniform(0.3, 0.7)), 2)),
                 o("achievement", "self", round(float(rng.uniform(0.5, 0.9)), 2))])
        e["session"] = f"t4p2-{seed}-{i}"
        p2.append(e)
    for i in range(32):
        pid = f"t4n-{seed}"
        e = exp(f"U8N-{seed:02d}-{i:02d}", ctx_of(i), ["collaboration", "music"], partner_list(pid, "collaborator"),
                [o("connection", pid, round(float(rng.uniform(0.8, 1.0)), 2)),
                 o("support", pid, round(float(rng.uniform(0.5, 0.9)), 2)),
                 o("achievement", "self", round(float(rng.uniform(0.4, 0.8)), 2))])
        e["session"] = f"t4n-{seed}-{i // 4}"
        nn.append(e)
    return p1, p2, nn


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
    # v3.1: 確率的な生成器の見本（乱数種 0）
    for k in "RKX":
        write_jsonl(f"gen_{k}_seed0.jsonl", generate_history(k, 0))
    write_jsonl("gen_praise_H1_seed0.jsonl", generate_praise_history("H1", 0))
    write_jsonl("gen_praise_H2_seed0.jsonl", generate_praise_history("H2", 0))
    # v3.2: ランダムな世界の見本、褒め言葉を伴う申し出の見本（乱数種 0）
    write_jsonl("gen_random_world_seed0.jsonl", generate_random_history(0))
    write_jsonl("gen_praise_offers_seed0.jsonl", generate_praise_offers(0))
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
