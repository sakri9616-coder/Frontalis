# FREEZE-v8: v8 の事前登録（問いの階段、改訂した段 3・4）の凍結（2026-10-22）

凍結するもの: `PREREG-v8-DRAFT.md` rev. a、`LADDER.md`（第 7 節まで）、測定・評価・判定のコード。SHA-256 は改行を LF に揃えて求めた（`python -B probes/p42_eval_v8.py hashes`）。評価のジョブは実行前に、`judge` は実行時と各 commit で照合する。この block はコードの `HASHED` とちょうど一致しなければならない。

```
cb97bc73fbfccdcd118fc1f369243064f307c3f80f55e379087155ae85053837 PREREG-v8-DRAFT.md
f3fc1c3480e918fea892cc69102b1ae97652e5537a0345e070642c572ce8a71b LADDER.md
0f11ee10442d508954b8bc71c33469965d824ca707304d7da7166b388efddd12 probes/sim4.py
aa21e4ba9f6f8cfa4f3e0053e7f08f74edc900178fc5df626ac7be1bad256b97 probes/measure4.py
66446ed35737930c660dd600014d782eb9ab70f3e7442d52b69bb9661916f608 probes/cells.py
f76360f18e602b2a52249f55126f01b60082e75aed36973e76e2b7df38e1791f probes/diag5/make_sim5.py
5d2b1140f42750fd12e823ec7bae30b8a9590d2184741aa48cd9691d08f7f73b probes/diag5/sim5.py
f36ccba9659dcf6a622f73136483a42533e39ac6cc36ae67ddb983000d877ae7 probes/p37_imprint.py
e8453071b7523f1282d750e24b0d6e0318a55778a25df8c50430567e94e4a224 probes/p39_ladder_dev.py
00f6c0402f0ac72aaa67d2ec9fe2d2325eb9a0ccb4af07b4ecd06998732820af probes/p41_ladder2_dev.py
b5fef53e3c00647b5ec6f31ba53dd752a56e9ba13067eb43a2bb8f02e495022d probes/p42_eval_v8.py
6ae0a070d73cfb9df14a5b0b6e249bc1a104b3c09423818b541e2bd75ce3f7ae probes/p15_lr.json
```

## 凍結の時点の事実
- 評価用の種は 77540〜77579（ユーザーが決めた）。まだ一度も実行していない。
- 凍結の門の査読は DeepSeek 1 本。回答は PREREG 第 7 節。
- `p42_eval_v8.py devcheck`（Colab T4、commit acfc186）で、開発用の種の P41 と全量が一致した（最大の差 0.0、`p42_devcheck.json`）。その後の変更は devcheck の終了コードと文書だけで、測定の関数は変えていない。
- 実行の計画: Colab の GPU で `p42_eval_v8.py job {77540,77550,77560,77570}`、`judge` で判定。結果はまずリーダーに報告する。
