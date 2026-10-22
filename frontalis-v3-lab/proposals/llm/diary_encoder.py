"""Diary -> NPN input (prototype v0, 2026-10-22; NOTES-llm-connection.md section 3). Not part of any preregistration.

The LLM writes what happened, WITHOUT good/bad words. This module turns those descriptions into the NPN's input format
(sim4.rows_np): entity tokens (48 = 32 feature dims + 16 identity dims, the identity dims are masked by the learner), context
tokens (16), and the outcome delta-b on the KB = 5 body channels.

Design choices (each one is meant to keep evaluation OUT of the converter):
- Text features use a hashing encoder (word and character-3-gram hashes -> fixed random signed projection -> tanh). It has no
  pretrained meaning, so it cannot carry a language model's sense of "nice" or "nasty". The cost: synonyms do not look alike.
  A pretrained embedding can replace it later only after the leakage check below passes.
- Descriptions are screened for evaluative words (a small Japanese/English lexicon). An entry that contains one is refused, not
  silently cleaned, so that the diary prompt can be fixed.
- The outcome is NOT a judgement of the event. It is the change of the agent's own body channels, read from observable facts
  with a fixed table (`CHANNELS`). This is the innate part, like pain or hunger: designed once, the same for every individual.
  Which persons and places predict those changes is what the NPN learns; that learned part is the evaluation.
- Who acted is often unknown in a diary; the converter keeps the pair and leaves attribution to the NPN.

    python llm/diary_encoder.py llm/sample_diary.json      -> prints the encoded stream and the screening report
"""
import hashlib, json, re, sys
from pathlib import Path
import numpy as np

D_FEAT, D_ID, D_CTX = 32, 16, 16           # sim4: D_E = 48 (32 features + 16 identity), D_C = 16
TOL = 0.4                                   # sim4 tolerance of the body channels (for reference: one level = 0.5 > TOL)
CHANNELS = (                                # KB = 5 body channels and the observable facts that move them (level -2..+2)
    ("energy", "体力・時間の余裕。長く待たされた、急かされた（−）／休めた（＋）"),
    ("safety", "身の安全。大声を向けられた、ぶつかられた（−）／守られた（＋）"),
    ("contact", "人とのつながり。話しかけられた、名前を呼ばれた（＋）／無視された、会話を切られた（−）"),
    ("knowing", "分かること。新しい情報を得た（＋）／説明が食い違って混乱した（−）"),
    ("progress", "自分の用事の進み。頼みが通った（＋）／止められた、やり直しになった（−）"),
)
LEVEL = 0.5                                 # one level of a channel = 0.5 in delta-b (sim4 outcomes are of order 1)
EVAL_WORDS = ("いい人", "良い人", "悪い人", "優しい", "やさしい", "親切", "意地悪", "嫌な", "嫌い", "好き", "最低", "最高", "ひどい",
              "感じのいい", "感じの悪い", "失礼", "怖い人", "素敵", "kind", "nice", "rude", "mean", "nasty", "good person", "bad person",
              "friendly", "hostile", "lovely", "awful")


def _proj(dim, salt):
    """A fixed random signed projection from 2^12 hash buckets to `dim`, the same on every machine."""
    g = np.random.default_rng(int(hashlib.sha256(salt.encode()).hexdigest()[:8], 16))
    return g.normal(size=(4096, dim)) / np.sqrt(64)


P_FEAT, P_CTX = _proj(D_FEAT, "frontalis-entity-v0"), _proj(D_CTX, "frontalis-context-v0")


def _buckets(text):
    t = text.lower()
    toks = re.findall(r"\w+", t) + [t[i:i + 3] for i in range(max(0, len(t) - 2))]
    v = np.zeros(4096)
    for tok in toks:
        h = int(hashlib.md5(tok.encode()).hexdigest()[:6], 16)
        v[h % 4096] += 1.0 if (h >> 12) % 2 == 0 else -1.0
    return v / (np.linalg.norm(v) + 1e-9)


def screen(text):
    return [w for w in EVAL_WORDS if w in text.lower()]


def encode_entity(name, looks):
    feat = np.tanh(_buckets(looks) @ P_FEAT * 4.0)
    ident = np.random.default_rng(int(hashlib.sha256(("id:" + name).encode()).hexdigest()[:8], 16)).normal(size=D_ID)
    return np.concatenate([feat, ident]).astype(np.float32)


def encode_context(place):
    return np.tanh(_buckets(place) @ P_CTX * 4.0).astype(np.float32)


def encode_outcome(levels):
    """levels: {channel name: integer -2..2} for the agent's own body; missing channels are 0."""
    bad = [k for k in levels if k not in [c for c, _ in CHANNELS]]
    if bad:
        raise ValueError(f"unknown channels {bad}")
    return np.array([LEVEL * int(np.clip(levels.get(c, 0), -2, 2)) for c, _ in CHANNELS], np.float32)


def encode_diary(diary):
    """diary: {"people": {name: looks}, "places": {name: description}, "events": [{"pair": [a, b], "place": p, "body": {...}}]}
    -> dict(world=dict(Xe, Xc), names, places, e1, e2, c, db, report)."""
    report = []
    for kind, table in (("people", diary["people"]), ("places", diary["places"])):
        for k, txt in table.items():
            hit = screen(txt)
            if hit:
                report.append(f"REFUSED {kind}:{k} contains evaluative words {hit}")
    names, places = list(diary["people"]), list(diary["places"])
    Xe = np.stack([encode_entity(n, diary["people"][n]) for n in names])
    Xc = np.stack([encode_context(diary["places"][p]) for p in places])
    e1, e2, c, db = [], [], [], []
    for i, ev in enumerate(diary["events"]):
        a, b = ev["pair"]
        if a not in names or b not in names or ev["place"] not in places:
            report.append(f"REFUSED event {i}: unknown person or place"); continue
        e1.append(names.index(a)); e2.append(names.index(b)); c.append(places.index(ev["place"])); db.append(encode_outcome(ev.get("body", {})))
    return dict(world=dict(Xe=Xe, Xc=Xc), names=names, places=places, e1=np.array(e1), e2=np.array(e2), c=np.array(c),
                db=np.stack(db) if db else np.zeros((0, len(CHANNELS)), np.float32), report=report, ok=not report)


def rows(enc, s, p, c):
    """The same layout as sim4.rows_np (self, partner, context, distractor zeros)."""
    W = enc["world"]
    return np.concatenate([W["Xe"][s], W["Xe"][p], W["Xc"][c], np.zeros((len(s), 8), np.float32)], 1)


if __name__ == "__main__":
    enc = encode_diary(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
    print("screening:", enc["report"] or "ok")
    print("people:", enc["names"]); print("places:", enc["places"])
    X = rows(enc, enc["e1"], enc["e2"], enc["c"])
    print("rows", X.shape, "(NPN D_IN = 120)"); print("delta-b per event:\n", enc["db"])
