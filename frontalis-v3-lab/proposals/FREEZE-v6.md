# FREEZE-v6: v6 の事前登録の凍結（2026-10-22）

凍結するもの: `PREREG-v6-DRAFT.md` rev. a と、その実行・判定のコード・設定。SHA-256 は、改行を LF に揃えて求めた（`python -B probes/p38_eval_v6.py hashes`）。

評価のジョブ（`p38_eval_v6.py job`）は、実行の前に下の一覧と一致するかを確かめ、違えば走らない（exit 3）。この block は、コードの `HASHED` とちょうど一致しなければならない。`judge` は、実行時にも再ハッシュし、ジョブが走ったすべての commit で、各ファイルが下の SHA と一致することを `git show` で確かめる。

```
169b092e002b616eb4629800bdcedf97c2d170bfc76df8ca3ba54ae961c000a2 PREREG-v6-DRAFT.md
0f11ee10442d508954b8bc71c33469965d824ca707304d7da7166b388efddd12 probes/sim4.py
aa21e4ba9f6f8cfa4f3e0053e7f08f74edc900178fc5df626ac7be1bad256b97 probes/measure4.py
66446ed35737930c660dd600014d782eb9ab70f3e7442d52b69bb9661916f608 probes/cells.py
f76360f18e602b2a52249f55126f01b60082e75aed36973e76e2b7df38e1791f probes/diag5/make_sim5.py
5d2b1140f42750fd12e823ec7bae30b8a9590d2184741aa48cd9691d08f7f73b probes/diag5/sim5.py
f36ccba9659dcf6a622f73136483a42533e39ac6cc36ae67ddb983000d877ae7 probes/p37_imprint.py
f377212631539dd5e2e644ba3c09ef4b61a22827f2519c6d92e9422bc68a687e probes/p38_eval_v6.py
6ae0a070d73cfb9df14a5b0b6e249bc1a104b3c09423818b541e2bd75ce3f7ae probes/p15_lr.json
651c4f538decd276b6423a36c3bd3eb22d47a9ea0a53f0381f667f01bdfe5fa5 probes/sched/modal_runner.py
```

## 凍結の時点の事実
- 評価用の種は 77440〜77479。まだ一度も実行していない。予備は 77500〜77539（別の事前登録でだけ使う）。
- 凍結の門の査読は DeepSeek 1 本。回答は PREREG 第 10 節。判定に nn0k4 を入れる案だけを退け、ほかは受け入れた。
- 開発用の種 77380〜77389 の試運転（`p38_dev_77380.json`、手元の RTX 3050）は、全腕を見ている。o3 は P37 と一致。nn0k4 も o3 に近い数値だった（PREREG 第 6 節に開示）。
- 実行の場所: Google Colab の GPU（ユーザーが有料プランを契約。2026-10-22）。4 ジョブを同じ機械の型で回す。
- 実行の計画: `python sched/sched.py run <queue> --only=colab`（待ち行列: `p38_eval_v6.py job {77440,77450,77460,77470}`）。併合と判定: `p38_eval_v6.py judge`。結果はまずリーダーに報告し、commit しない。
