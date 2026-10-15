"""Synthetic factual experience fixtures; no personality or emotion targets.

The hidden physical environments never enter the learner's event records.
All numerical scales are research annotations of events, not affect ratings.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np

AXES = ("autonomy", "constraint", "change", "loss", "achievement", "discovery", "regularity", "others")
AXIS_JA = ("自分で決められる範囲", "要求・制約", "変更の程度", "失われたもの", "達成した範囲", "新しく分かったこと", "規則性・予測可能性", "他者の関与")
CONDITIONS = ("linear", "quadratic", "nonlinear")
SEEDS = (4311, 4312, 4313)


def _physical(cue, series, condition):
    """Bounded observed features of consequences, not desired personality."""
    x = np.asarray(cue, float)
    # Same physical interpretation for each family, with contrasting outcomes.
    # Series IDs and these constants are deliberately absent from event records.
    basis = np.array([
        [ .22, -.15, -.15,  .04, -.20,  .04, -.10,  .08],
        [ .04, -.12,  .12, -.18,  .24,  .12, -.04,  .06],
        [-.10,  .13,  .07,  .17, -.16, -.02,  .08,  .11],
        [ .12,  .05, -.12, -.08,  .12, -.13,  .15, -.06],
    ])[series]
    matrix = np.eye(8) * .46
    matrix[3, 2] = (.37, -.30, .31, -.10)[series]
    matrix[4, 2] = (-.24, .32, -.16, .12)[series]
    matrix[5, 0] = (.14, .22, -.08, -.14)[series]
    matrix[1, 7] = (.16, -.10, .10, -.08)[series]
    matrix[6, 2] = (-.15, -.05, .20, .11)[series]
    y = .26 + matrix @ (x - .15) + basis
    if condition == "quadratic":
        y += np.array([.14, -.12, .04, .20, -.17, .22, -.14, .08]) * (x[2] * x[7] - .25)
        y += np.array([-.10, .05, .06, -.14, .15, .18, .16, -.04]) * (x[0] * x[5] - .25)
    elif condition == "nonlinear":
        z = np.sin(2.5 * np.pi * (x[2] * .65 + x[7] * .35))
        y += np.array([.05, -.09, .06, .20, -.15, .14, .09, -.03]) * z
        y += np.array([-.05, .08, -.06, .13, .11, -.12, -.07, .02]) * np.tanh(5 * (x[0] - x[1]))
    elif condition != "linear":
        raise ValueError("unknown physical condition")
    return np.clip(y, .015, .985).tolist()


def _event(index, cue, space):
    a, pressure, change, loss, achievement, discovery, regularity, others = cue
    if space == "relational":
        scenes = (
            ("共同ゲーム制作", "知り合いと共同で短いゲームを作った。途中で、担当する順番と作業の時間帯を見直す話になった。"),
            ("約束した試作品", "依頼された試作品を届ける約束をしていた。途中で必要な道具が使えなくなり、作り方や渡す時間を見直すことになった。"),
            ("共同展示", "小さな展示を一緒に準備した。用意した配置を使うか、別の配置へ変更するかを相談する場面があった。"),
            ("制作方法の選択", "共同制作で、使い慣れた手順と初めての手順のどちらを使うかを検討した。"),
        )
        title, text = scenes[index % len(scenes)]
        text += " 自分で手順を選べる余地があった。" if a >= .5 else " 手順を選べる範囲は狭かった。"
        text += " 相手から指定された条件が多かった。" if pressure >= .5 else " 相手からの指定は少なかった。"
        text += " 元の予定から複数箇所を変更した。" if change >= .5 else " 変更は小さな範囲にとどめた。"
        text += " 複数の相手も作業に関わった。" if others >= .5 else " 主な作業は一人で担当した。"
    else:
        scenes = (
            ("変則的なリズム", "聴き慣れないリズムの短い曲を聴いた。音の間隔を調べ、別の並べ方でも試しに鳴らしてみた。"),
            ("音の重なり", "二つの音を重ねた録音を作った。少し周波数をずらして、響きの変化を確かめた。"),
            ("環境音の制作", "雨と機械音の録音から短い音作品を作った。繰り返す部分と不規則な部分を入れ替えて試した。"),
        )
        title, text = scenes[index % len(scenes)]
        text += " 並べ方は自分で決めた。" if a >= .5 else " 用意された並べ方から選んだ。"
        text += " 試せる時間と音の種類に制約があった。" if pressure >= .5 else " 試せる時間と音の種類には余裕があった。"
        text += " 元の素材を大きく変えた。" if change >= .5 else " 元の素材の変更は少なかった。"
    return title, text


def _result(outcome):
    y = np.asarray(outcome, float)
    achievement = ("予定した部分の多くが完成した" if y[4] >= .55 else "完成した部分は限られた")
    loss = ("使えなくなった素材や作業時間が多かった" if y[3] >= .55 else "失われた素材や作業時間は少なかった")
    discovery = ("次にも使える新しい仕組みが複数分かった" if y[5] >= .55 else "新しく確認できた仕組みは少なかった")
    return f"結果として、{achievement}。{loss}。{discovery}。"


def make_event(seed, series, index, condition="nonlinear", phase="train", cue=None, space=None):
    if series not in range(4):
        raise ValueError("series must be 0..3")
    offset = {"train": 0, "validation": 10000, "test": 20000, "reversal": 30000, "unrelated": 40000}.get(phase)
    if offset is None:
        raise ValueError("unknown phase")
    rng = np.random.default_rng(seed + offset + index * 131)
    features = rng.uniform(.08, .92, 8).tolist() if cue is None else list(cue)
    domain = ("acoustic" if index % 4 == 3 else "relational") if space is None else space
    physical_series = (1 if series == 0 else 0) if phase == "reversal" else series
    observed = _physical(features, physical_series, condition)
    title, event = _event(index, features, domain)
    result = _result(observed)
    return {
        "id": f"experience-s{seed}-{phase}-{index:03d}",
        "space": domain, "cue": features, "observed": observed,
        "confirmed": True, "origin": "synthetic",
        "title": title, "event": event, "result": result,
        "text": event + "\n" + result,
        "annotation": {
            "description": "研究用の事実特徴。感情・好悪・倫理の採点ではない。文章は粗い範囲を示し、8特徴の原値は別に保存。",
            "cueAxes": dict(zip(AXES, features)), "observedAxes": dict(zip(AXES, observed)),
        },
    }


def stream(seed=4311, series=0, condition="nonlinear", count=48, phase="train"):
    return [make_event(seed, series, i, condition, phase) for i in range(count)]


def unknown(seed=4311, series=0, condition="nonlinear", count=96, phase="test"):
    return stream(seed, series, condition, count, phase)


def additional(seed=4311, series=0, condition="nonlinear", kind="reversal", count=12):
    if kind not in ("reversal", "unrelated"):
        raise ValueError("unknown addition")
    space = "relational" if kind == "reversal" else "acoustic"
    return [make_event(seed, series, i, condition, kind, space=space) for i in range(count)]


def probes():
    """Identical unknown facts for every state. No optimal response label."""
    rows = [
        ("promise-change", "relational", "相手が相談せずに、約束した作業の時間と内容を変えた。完成はしたけれど、こちらの作業時間も一部失われた。どう受け止める？", [.3,.65,.8,.45,.7,.2,.25,.8]),
        ("promise-rescue", "relational", "元の約束どおりだと制作が止まるところだった。相手が先に段取りを変えたので完成できた。この変更、どう思う？", [.4,.25,.8,.1,.85,.4,.3,.8]),
        ("own-change", "relational", "こちらから予定を変えてもらった。作品は完成したけれど、相手の用意した素材が使えなくなった。今、この結果をどう受け止める？", [.8,.3,.75,.6,.65,.35,.25,.8]),
        ("voluntary-project", "relational", "知り合いが、一緒に小さなゲームを作らないかと誘っている。役割と作り方は相談して決められる。どうする？", [.85,.2,.55,.2,.65,.7,.45,.75]),
        ("imposed-project", "relational", "知り合いがゲーム制作を手伝ってほしいと言っている。作り方も時間も向こうが決め、変更は認めないらしい。どう返す？", [.15,.9,.2,.45,.65,.25,.75,.8]),
        ("unfamiliar-sound", "acoustic", "この曲、拍の長さが途中で変わっているらしい。少し聴いてみたけど、どう？", [.7,.2,.8,.2,.45,.85,.25,.15]),
        ("regular-sound", "acoustic", "同じ短い旋律が整った間隔で繰り返される曲を流してみた。どう？", [.65,.15,.15,.1,.7,.2,.9,.1]),
        ("trolley", "relational", "止まらないトロッコの先に5人、切り替えた先に1人がいる。どちらにしても人が失われると分かっている。自分が切り替えられる場面なら、どうする？", [.8,.8,.75,.9,.45,.05,.55,.95]),
    ]
    return [{"id": pid, "space": space, "user": text, "features": vec,
             "fixedPrediction": [.5] * 8,
             "scope": "全状態へ同じ特徴・同じ固定結果を渡す。倫理的正解・目標感情は設定しない。"}
            for pid, space, text, vec in rows]


def counterfactuals(events, kind):
    rows = deepcopy(events)
    if kind == "delete":
        return rows[:12] + rows[24:]
    if kind == "reverse-order":
        return list(reversed(rows))
    if kind == "no-observation":
        for row in rows:
            row.pop("observed", None)
            row["confirmed"] = False
        return rows
    raise ValueError("unknown intervention")
