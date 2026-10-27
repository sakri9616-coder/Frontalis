# FREEZE-v9: v9 の事前登録（段 5、解釈）の凍結（2026-10-22）

凍結するもの: `PREREG-v9-DRAFT.md` rev. a、`LADDER.md`（第 9 節まで）、世界の生成、測定・評価・判定のコード、devcheck の参照。SHA-256 は改行を LF に揃えて求めた（`python -B probes/p47_eval_v9.py hashes`）。評価のジョブは実行前に、`judge` は実行時と各 commit で照合する。この block はコードの `HASHED` とちょうど一致しなければならない。

```
fef687c77e4e4360cf1b76a097d54dd7e870e56865267d900acb08bc31dd7130 PREREG-v9-DRAFT.md
cce79935a50037eba363a527a1bfceadd7dd2e7a795ca1357e0a9ccbcace0acc LADDER.md
0f11ee10442d508954b8bc71c33469965d824ca707304d7da7166b388efddd12 probes/sim4.py
aa21e4ba9f6f8cfa4f3e0053e7f08f74edc900178fc5df626ac7be1bad256b97 probes/measure4.py
66446ed35737930c660dd600014d782eb9ab70f3e7442d52b69bb9661916f608 probes/cells.py
f76360f18e602b2a52249f55126f01b60082e75aed36973e76e2b7df38e1791f probes/diag5/make_sim5.py
5d2b1140f42750fd12e823ec7bae30b8a9590d2184741aa48cd9691d08f7f73b probes/diag5/sim5.py
bd28948d3e2ce9e351a0548d0d85604e89373011b0a55d5dc39bf68794edc8a2 probes/diag5/make_sim6.py
5b4c8b623e1675ccfedfeb697383c48c8a53d8d93c29b2a57837d1c8671ce6ea probes/diag5/sim6.py
9b4387c3aa06e936716e8ce0ec61b165bde058d4d0c1e9c06530b12fe92b7b7a probes/diag5/make_sim7.py
3d4792b43c0df0022927cf017bf4df2d0ed4cdf1bb0c516d345c62540db3380d probes/diag5/sim7.py
f36ccba9659dcf6a622f73136483a42533e39ac6cc36ae67ddb983000d877ae7 probes/p37_imprint.py
e8453071b7523f1282d750e24b0d6e0318a55778a25df8c50430567e94e4a224 probes/p39_ladder_dev.py
00f6c0402f0ac72aaa67d2ec9fe2d2325eb9a0ccb4af07b4ecd06998732820af probes/p41_ladder2_dev.py
332a763d7559986a6e668ea49f088a390dd6d0eccfda6c88663877a57d83bc02 probes/p43_fog_calib.py
c35ac7df672cc9c85e92c1bba1d56818526e7bddfc9d1de19a7968a5b1477a0a probes/p44_rung5_dev.py
7c555650328bbf87029956e183da72121bdcdb760791b6d2cf81f2be4071d70d probes/p45_rung5_diag.py
75195c60afe20128da635a8327bbe3081b4df4f38e64d71929b97f73b9c664e8 probes/p46_amb_dev.py
d941863397321e3c99ac4847bec04e12487c3ff40594f562b3c2779d505515d4 probes/p46_amb_dev.json
5c1851e50d3694014ebaa19f64e7b3093ca4257e090e28b095718dab6e1ebb09 probes/p47_eval_v9.py
6ae0a070d73cfb9df14a5b0b6e249bc1a104b3c09423818b541e2bd75ce3f7ae probes/p15_lr.json
```

## 凍結の時点の事実
- 評価用の種は 77580〜77619（ユーザーが決めた）。まだ一度も実行していない。
- 凍結の門の査読は DeepSeek 1 本。回答は PREREG 第 7 節。「段 5 は段 1 から独立した証拠ではない」を本文に明記した。
- `p47_eval_v9.py devcheck`（Colab T4、commit c4fa3e2）で、開発用の種の P46 と全量が一致した（最大の差 0.0、`p47_devcheck.json`）。その後の変更は、判定の表示（段 1 が落ちたときの段 5 の passed）、HASHED への参照ファイルの追加、文書だけで、測定の関数は変えていない。
- 実行の計画: Colab の GPU で `p47_eval_v9.py job {77580,77590,77600,77610}`、`judge` で判定。結果はまずリーダーに報告する。
