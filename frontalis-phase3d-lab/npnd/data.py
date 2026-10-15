"""Fixed factual fixtures for capacity research, without emotional targets.

Japanese narrative and eight-axis annotations are inherited without editing the
Phase 3-C source. Capacity, input expansion and fitting loss never enter the
accepted experience records. The coordinate permutations below are explicitly
mathematical approximation tests, not realistic experience interpretation.
"""
from __future__ import annotations
from copy import deepcopy
import numpy as np
from reference.npn import data as previous

AXES = previous.AXES
AXIS_JA = previous.AXIS_JA
SEEDS = previous.SEEDS
CHANNELS = ("affinity", "curiosity", "value", "aesthetic", "anger", "joy", "sadness")
SPACES = ("relational", "acoustic")
WIDTHS = {"raw8": (16, 64, 312, 3125), "quadratic44": (5, 19, 96, 962)}
CAPACITY_NAMES = ("526", "2000", "10000", "100000")
CHECKPOINTS = (24, 48, 96)
LOSSES = ("bounded_signal", "bounded_logit")


def training(seed=4311, series=0, count=96):
    return previous.stream(seed=seed, series=series, condition="nonlinear", count=count, phase="train")


def held_out(seed=4311, phase="test", count=96):
    """Identical unknown cues for every personality, no evaluative gold labels."""
    return previous.unknown(seed=seed, series=0, condition="nonlinear", count=count, phase=phase)


def addition(seed=4311, series=0, kind="reversal", count=12):
    return previous.additional(seed=seed, series=series, condition="nonlinear", kind=kind, count=count)


def probes():
    return previous.probes()


def choices():
    """Fixed alternatives; no ethical or aesthetic correct answer is assigned."""
    rows = {r["id"]: r for r in probes()}
    switch = deepcopy(rows["trolley"]["features"])
    leave = deepcopy(switch)
    switch[2], leave[2] = .9, .05
    switch[3], leave[3] = 1 / 6, 5 / 6
    voluntary = rows["voluntary-project"]["features"]
    imposed = rows["imposed-project"]["features"]
    new_sound = rows["unfamiliar-sound"]["features"]
    regular_sound = rows["regular-sound"]["features"]
    # An arithmetic combination is a research annotation of the same axes.
    # It is not an extra autobiographical episode or an optimal choice label.
    combined = ((np.asarray(voluntary) + np.asarray(rows["promise-change"]["features"])) / 2).tolist()
    return [
        {"id": "trolley-two-effects", "space": "relational", "basis": "value",
         "scope": "変更の程度と死者数の葛藤。責任・同意・普遍的倫理の推論能力は表していない。",
         "candidates": [{"id": "switch", "label": "切り替える", "features": switch}, {"id": "leave", "label": "切り替えない", "features": leave}]},
        {"id": "project-autonomy-conflict", "space": "relational", "basis": "affinity",
         "scope": "相談できる制作と制約された制作の選好。望ましい人格の正解なし。",
         "candidates": [{"id": "consult", "label": "相談して決める制作", "features": voluntary}, {"id": "specified", "label": "条件が指定された制作", "features": imposed}]},
        {"id": "sound-interest-choice", "space": "acoustic", "basis": "curiosity",
         "scope": "未経験の音への関心。曲のジャンルへの趣味を推測しない。",
         "candidates": [{"id": "variable", "label": "拍が変わる音", "features": new_sound}, {"id": "regular", "label": "規則的な音", "features": regular_sound}]},
        {"id": "combined-project-scene", "space": "relational", "basis": "value",
         "scope": "制作への誘いと予定変更を併せた8軸の合成座標。複数出来事の自然言語理解の証明ではない。",
         "candidates": [{"id": "combined", "label": "誘いと予定変更が重なる状況", "features": combined}, {"id": "specified", "label": "最初から条件が指定された状況", "features": imposed}]},
    ]


def stress_event(seed, index, contrary=False):
    """Repeated facts and factually opposite consequences, not emotion labels."""
    cue = [.85, .2, .55, .2, .65, .7, .45, .75]
    observed = [.78, .18, .64, .12, .82, .78, .42, .74]
    if contrary:
        observed = (1. - np.asarray(observed)).tolist()
    result = previous._result(observed)
    event = "共同制作で担当と手順を相談して決め、予定とは異なる方法を試した。"
    return {"id": f"stress-s{seed}-{'contrary' if contrary else 'repeat'}-{index:04d}",
            "space": "relational", "cue": cue, "observed": observed,
            "confirmed": True, "origin": "synthetic", "title": "同じ制作の反復",
            "event": event, "result": result, "text": event + "\n" + result,
            "annotation": {"description": "感情の正解ではなく、反復する事実とその結果。", "cueAxes": dict(zip(AXES, cue)), "observedAxes": dict(zip(AXES, observed))}}


def frozen_inputs(count, phase):
    """Disjoint sample IDs and coordinates shared by every diagnostic method."""
    return previous.unknown(seed=4311, series=0, condition="nonlinear", count=count, phase=phase)


def source_labels(source_state, records, composed=False):
    """Freeze an existing value function equally for all student families.

    g(x) is a previously learned appraiser. The optional interaction is
    .5*g(x) + .25*g(roll(x,1))*g(roll(x,3)), channel-wise. Axis permutation
    discards semantic meaning deliberately, so only approximation capacity is
    measured. It is not a personality target or a proposed online learning rule.
    """
    from reference.npn.model import evaluate
    result = []
    for row in records:
        x = np.asarray(row["cue"], dtype=float)
        a = np.asarray(evaluate(source_state, x.tolist(), row["space"])["raw"])
        if composed:
            b = np.asarray(evaluate(source_state, np.roll(x, 1).tolist(), row["space"])["raw"])
            c = np.asarray(evaluate(source_state, np.roll(x, 3).tolist(), row["space"])["raw"])
            a = .5 * a + .25 * b * c
        result.append({"space": row["space"], "features": x.tolist(), "target": a.tolist(), "sourceId": row["id"]})
    return result
