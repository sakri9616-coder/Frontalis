# FREEZE-v7: v7 の事前登録（問いの階段、段 1〜4）の凍結（2026-10-22）

凍結するもの: `PREREG-v7-DRAFT.md` rev. a、`LADDER.md`（第 6 節まで）、測定・評価・判定のコード。SHA-256 は改行を LF に揃えて求めた（`python -B probes/p40_eval_v7.py hashes`）。評価のジョブは実行前に、`judge` は実行時と、ジョブが走った各 commit で、下の一覧と照合する。この block はコードの `HASHED` とちょうど一致しなければならない。

```
9952c33fcb949cef90c5b6a807977972f74ad3228ba402e536ea683a25912b9d PREREG-v7-DRAFT.md
f8c66bf76af754560748419decc2256ff99e66d78057e7fa148d93899b863e72 LADDER.md
0f11ee10442d508954b8bc71c33469965d824ca707304d7da7166b388efddd12 probes/sim4.py
aa21e4ba9f6f8cfa4f3e0053e7f08f74edc900178fc5df626ac7be1bad256b97 probes/measure4.py
66446ed35737930c660dd600014d782eb9ab70f3e7442d52b69bb9661916f608 probes/cells.py
f76360f18e602b2a52249f55126f01b60082e75aed36973e76e2b7df38e1791f probes/diag5/make_sim5.py
5d2b1140f42750fd12e823ec7bae30b8a9590d2184741aa48cd9691d08f7f73b probes/diag5/sim5.py
f36ccba9659dcf6a622f73136483a42533e39ac6cc36ae67ddb983000d877ae7 probes/p37_imprint.py
e8453071b7523f1282d750e24b0d6e0318a55778a25df8c50430567e94e4a224 probes/p39_ladder_dev.py
734b281a6bc3c347ff377eaa562cb659d9870b3e423663583de5d71efa328e51 probes/p40_eval_v7.py
6ae0a070d73cfb9df14a5b0b6e249bc1a104b3c09423818b541e2bd75ce3f7ae probes/p15_lr.json
```

## 凍結の時点の事実
- 評価用の種は 77500〜77539。まだ一度も実行していない。これで予約の範囲は尽きる。
- 凍結の門の査読は DeepSeek 1 本。回答は PREREG 第 7 節。
- `p40_eval_v7.py devcheck`（Colab T4、commit 8f41eef）で、開発用の種の P39 の結果と、全量が完全に一致した（最大の差 0.0、`p40_devcheck.json`）。
- 実行の計画: Colab の GPU で `p40_eval_v7.py job {77500,77510,77520,77530}`、`judge` で判定。結果はまずリーダーに報告する。
