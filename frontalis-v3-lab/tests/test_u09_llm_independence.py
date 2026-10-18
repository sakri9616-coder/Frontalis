"""U9 LLM からの独立: 埋め込みは NPN の状態が所有する。

SPEC-v3.1 第 3 節: 状態を保存した後、lexicon を別の座標（回転したベクトル）に替えて復元し、既知の類・概念を評価する。
合格: 評価が完全に一致する（埋め込みは状態が所有している）。

測り方（実装前に固定した細部）:
  乱数種ごとに R・K・X の人格（型は種に応じて巡回）を育てて保存し、lexicon の全ベクトル（類・文脈・概念）に種ごとのランダムな直交行列を掛けた
  別の座標の lexicon で復元する。その人格が経験で登録済みの類・文脈・概念だけを使う評価（新しい経験の appraise、状況・エンティティ・概念・手がかりの
  プローブ）が、元の NPN と完全に一致する（deep_equal、許容誤差なし）。
  対照: 状態が埋め込みを所有しない（v3.0 のように lexicon を読む）なら一致しないこと — 回転した lexicon で新しく作った NPN が、
  同じ経験史から違う状態になることで、回転が実際に評価を変える設定だったことを確かめる。
  さらに、保存ファイルに lexicon の中身は入らない（所有する埋め込みの値だけが入る）。
"""
import json
import os
import tempfile
import unittest

import numpy as np

from helpers import KINDS, SEEDS, deep_equal, dump, make, mh, persona_gen
from npnv3 import Lexicon, NPN



def known_words(n):
    """その人格が経験で登録済みの語（類・文脈・概念）。U9 の評価は、既知の語だけを使う。"""
    return (next(iter(n.emb.own["cls"])), next(iter(n.emb.own["ctx"])), list(n.emb.own["concept"])[:2])


def evaluate(n):
    cls, ctx, concepts = known_words(n)
    exp = {"id": "u9-new", "context": ctx, "concepts": concepts, "participants": [{"id": "u9-friend", "class": cls}],
           "outcomes": [{"type": "connection", "actor": "u9-friend", "magnitude": 0.8},
                        {"type": "achievement", "actor": "self", "magnitude": 0.6}]}
    probes = [{"context": ctx, "concepts": concepts, "participants": [{"id": "u9-friend", "class": cls}]},
              {"entity": "u9-friend", "class": cls, "context": ctx}, {"entity": "u9-friend", "class": cls},
              {"concept": concepts[0]}, {"concept": concepts[1]}, {"cue": "type:connection"}]
    return [n.appraise(exp)] + [n.probe(p) for p in probes]


def rotation(seed):
    q, _ = np.linalg.qr(np.random.default_rng([seed, 9]).normal(size=(8, 8)))
    return q


class U9LLMIndependence(unittest.TestCase):
    def test_swapped_lexicon_does_not_change_known_evaluations(self):
        src = Lexicon.load()
        matched = 0
        for seed in SEEDS:
            kind = KINDS[seed % 3]
            n = persona_gen(kind, seed)
            rot = src.rotated(rotation(seed))
            with tempfile.TemporaryDirectory() as d:
                n.save(d)
                with open(os.path.join(d, "state.json"), encoding="utf-8") as f:
                    text = f.read()
                self.assertNotIn("dims", json.loads(text)["meta"])                  # lexicon の写しは保存しない
                restored = NPN.load(d, lexicon=rot)
            self.assertTrue(deep_equal(evaluate(n), evaluate(restored)), f"seed {seed}")
            matched += 1
            # 対照: 回転した lexicon で最初から育てると、状態は違う（回転は評価を実際に変える）
            fresh = make(seed, lexicon=rot)
            for e in mh.generate_history(kind, seed, 36):
                fresh.experience(e)
            self.assertFalse(deep_equal(evaluate(n)[0]["Q"], evaluate(fresh)[0]["Q"]))
        dump("U09", {"seeds_exactly_equal_after_lexicon_swap": matched, "n_seeds": len(SEEDS)})

    def test_new_words_after_restore_use_the_new_lexicon(self):
        src = Lexicon.load()
        n = persona_gen("K", 0)
        rot = src.rotated(rotation(0))
        with tempfile.TemporaryDirectory() as d:
            n.save(d)
            restored = NPN.load(d, lexicon=rot)
        self.assertNotIn("family", n.emb.own["cls"])
        e = {"id": "new-word", "context": "cafe_chat", "concepts": ["learning"],
             "participants": [{"id": "m1", "class": "family"}], "outcomes": [{"type": "comfort", "actor": "m1", "magnitude": 0.6}]}
        restored.experience(e)
        self.assertTrue(np.allclose(restored.emb.own["cls"]["family"], rot.lookup("cls", "family")))
        self.assertTrue(np.allclose(restored.emb.own["cls"]["collaborator"], src.lookup("cls", "collaborator")))   # 既知の語は元の座標のまま


if __name__ == "__main__":
    unittest.main()
