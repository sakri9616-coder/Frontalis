# FREEZE: 主の経路（最小の観測）の凍結

- 凍結: 2026-10-17、commit **a07a608**（ブランチ claude/value-representation-design）。PREREG 草案 4 rev. b2 の第 10d 節（主の経路）と、それが参照する節（第 4〜8 節の定義・判定・較正・結論の規則、第 3 節の種）。
- 根拠: ユーザーの判断（リーダー経由）。前提の監査は批評 1〜3 と 4 つの独立の査読（DeepSeek）で満たされたとし、査読への回答（DESIGN 第 14.6 節）を反映した後に凍結する。
- SHA-256 は、commit a07a608 に入っている内容（`git show a07a608:<path>`、改行は LF）から計算した。作業ツリーの改行が CRLF に変わっている場合は、`git show` の出力で照合する。

## 凍結したファイル（frontalis-v3-lab/proposals/ からの相対パス）
```
19c7d236dc48db0139bec0748a2418ee595f2babf4d790ef693886ad116e2973  PREREG-DRAFT.md
8afee1cbf883c5f16671a4641c8e231abca042e11715ce8d26a8f9404cafba15  DESIGN.md
0f11ee10442d508954b8bc71c33469965d824ca707304d7da7166b388efddd12  probes/sim4.py
aa21e4ba9f6f8cfa4f3e0053e7f08f74edc900178fc5df626ac7be1bad256b97  probes/measure4.py
66446ed35737930c660dd600014d782eb9ab70f3e7442d52b69bb9661916f608  probes/cells.py
6b704365fb5071c3e8627a159e526fea04f4fd10561e9b2f098595682171f194  probes/judge4.py
bcab2aa60fa3c046ef4179ccf0f77e4cf5ee16e4740b24e98c963ab21fdcac20  probes/p30_eval_main.py
8a8eda2bf12df143a63d561abb0838141007bb2fa31f88ebd158bfd9ae88ffa2  probes/p25_min_dev.py
b6cffd16dec41de8fff87fbc7f9a479cb01ce4902fd213c1c3c80a9fcf3d0009  probes/p15_prod.py
992b71c2f803b6260d7b6d5c92da1141308df0f4d3681f89a798ce6b84194b75  probes/p13_ckpt_early.py
6ae0a070d73cfb9df14a5b0b6e249bc1a104b3c09423818b541e2bd75ce3f7ae  probes/p15_lr.json
f105e72cbb6f0fa71b12268444949ede2e23482fe2d731d08cf51035c6e89685  probes/p16_pscale.json
44bd0f8d0cc795221b49addda94159e50015256834c62229e1fce8b3a8483eda  probes/p19_results.json
25c252e92884bfd875dc4ffaee7f619e55cfefb766fb69506c506cbbcc930ef0  probes/p30_calib_coff.json
```

## 評価の実行（リーダーの開始の合図の後、手元の 1 か所で、1 回だけ）
1. 照合: 上の SHA-256 が `git show a07a608:frontalis-v3-lab/proposals/<path> | sha256sum` と一致すること。
2. 評価の前の開発用の結果がそろっていること: `probes/p25_calib_slp.json`（C-SLp の較正）、`probes/p15_cells_nn0_IH.json`、`probes/p15_cells_krr_IH.json` の C-SLp。これらは凍結した `p25_min_dev.py`・`p15_prod.py` で作り、結果は第 6 節の規則で機械的に使う（使えない対比は使わない。直さない）。
3. `python -B p13_ckpt_early.py`（決定性と再開。identical: True）。
4. `python -B p30_eval_main.py train`（評価用の種 77100〜77139。区画はチェックポイントから自動で再開する。計算の故障で止まった場合は同じ設定で再開し、回数を報告する）。
5. `python -B p30_eval_main.py judge` → `p30_judgment.json`。
6. 結果はリーダーに先に報告し、公開しない（commit・push しない）。

## 環境（手元）
Windows 11、Python 3.12.13、torch 2.7.0+cu128（CUDA 12.8、cuDNN 9.7.1）、numpy 2.5.4、NVIDIA GeForce RTX 3050（VRAM 8 GB、他のエージェントと共有）、ホストの RAM 32 GB。

## 凍結の後に変えてよいもの・いけないもの
- 変えてはいけない: 上のファイル。判定の規則、下限、種、セル、対照、較正の使い方。
- 変えてよい: 記録（`probes/LOG-dev.md`、`README.md`）、任意の段の実行（PREREG 第 10d 節 7 の引き金が成り立ったときだけ、種 77400〜77439）。
- コードの変更が要る失敗が評価の途中で起きた場合: 評価を止め、変更を記録し、変更の前の区画の結果を残し、リーダーに報告する（PREREG 第 10 節）。
