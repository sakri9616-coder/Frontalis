# FREEZE-v5: v5 の事前登録の凍結（2026-10-18）

凍結するもの: `PREREG-v5-DRAFT.md` rev. b と、その実行・判定のコード・較正・設定。SHA-256 は、作業の複製のファイルの改行を LF に揃えて求めた（`python -B probes/p33_eval_v5.py hashes`）。

評価のジョブ（`p33_eval_v5.py job`）は、実行の前に下の一覧と一致するかを確かめ、違えば走らない。併合（`check`）は、ジョブが走ったすべての commit で、この一覧のファイルが同じバイト列かを `git show` で確かめる。

```
21225a0a1b5ac72d2fd4e01e1880717a378afe49e6b42027c8afe0185316c98f PREREG-v5-DRAFT.md
0f11ee10442d508954b8bc71c33469965d824ca707304d7da7166b388efddd12 probes/sim4.py
aa21e4ba9f6f8cfa4f3e0053e7f08f74edc900178fc5df626ac7be1bad256b97 probes/measure4.py
66446ed35737930c660dd600014d782eb9ab70f3e7442d52b69bb9661916f608 probes/cells.py
6b704365fb5071c3e8627a159e526fea04f4fd10561e9b2f098595682171f194 probes/judge4.py
e099fc8dc6ee619a5f8335ea1b0d0b03d051562d6786c1bf41b4f8930afcfccb probes/measure5.py
f76360f18e602b2a52249f55126f01b60082e75aed36973e76e2b7df38e1791f probes/diag5/make_sim5.py
5d2b1140f42750fd12e823ec7bae30b8a9590d2184741aa48cd9691d08f7f73b probes/diag5/sim5.py
736ef43b2bab3aac96bdaf4368ffa309344c1e2ceab2e280aa66d31cb8638ac2 probes/p32_v5_dev.py
cbd12828b7dfa945f4d5c4fc63d50d400b81246a890fec54599e16e1f0515d86 probes/p33_eval_v5.py
3bb7c41e03d26f1e2929061c45756013694af5f59bfab68703dc2736005ed4ff probes/p32_setwg_modal.json
5d9a28eaeace1404d8f4b6be64ce7678cb3d384ac6725648bf04af7157403b12 probes/p33_v5_constants.json
caa39bff4a5e219279ba810b83dfcc7a0dd4c0606dc1fb22f5c0f6fc95269632 probes/p33_v5_config.json
6ae0a070d73cfb9df14a5b0b6e249bc1a104b3c09423818b541e2bd75ce3f7ae probes/p15_lr.json
f105e72cbb6f0fa71b12268444949ede2e23482fe2d731d08cf51035c6e89685 probes/p16_pscale.json
784174ac297ab77985441fdb7afb1d0b6dfc0079a46a4e10400506d60b996bdf probes/sched/sched.py
651c4f538decd276b6423a36c3bd3eb22d47a9ea0a53f0381f667f01bdfe5fa5 probes/sched/modal_runner.py
```

## 凍結の時点の事実
- 評価用の種は 77300〜77339。まだ一度も実行していない。`sched/sched.py` は、`--allow-eval` なしではこの範囲を拒む（plan で確認済み）。
- 草案 4 と共有するファイル（sim4、measure4、cells、judge4、p15_lr、p16_pscale）は、`FREEZE.md` のハッシュと同じである。
- J3v5 の定数は `p33_v5_constants.json` にある。評価と同じ型の機械（Modal T4、torch 2.14.1+cu130、commit 818155d）の較正から、事前の規則で求めた: m_abs 0.020、J3v5 は使える。
- 草案 4 の評価のジョブ 32 件は終わっている。その結果は、筆者は開いていない。v5 の設計と凍結は、その結果を見ずに行った。

## 開発の結果を見た後の変更（2026-10-18。PREREG-v5 第 12 節）
- V2 を、ユーザーの判断で「学習した予期が、未見の文脈へ移る」に言い換えた。
  - 合格の条件は、C-SLp で J5-a（R²_nc > 0）と J5-c_nn0（NN-0 に勝つ）。
  - 帰属を揃えた KRR との比較は、報告だけにした。
- **この変更は、開発用の種の結果を見た後に行った。**
  - 見ていた結果: KRR の R²_nc 0.640。NN-SET は 0.578 ／ 0.572 で、KRR を上回ったのは 10 種中 2 種。NN-SET は NN-0 に 10/10 で勝つ。
  - 旧 V2 は通らない見込みで、新 V2 は通る見込みである。
- 理由（ユーザー）:
  - 周期の帰属を与えた KRR との正確さの競争は、人格の形成について何も言わない。
  - 評価の土台として大事なのは、行為の帰結についての予期であって、世界に対する正確さではない。
- この理由が、開発の結果と独立だったとは主張しない。時期と理由の両方をここに記す。

## 実行の計画（まだ始めていない）
- 実行器: `python sched/sched.py run <queue> --allow-eval --only=modal`（Modal T4 だけ）。
- 待ち行列: `p33_eval_v5.py job {set,krr,nn0} {77300,77305,...,77335}`（24 ジョブ。各 1 つの結果ファイル）。
- 併合と判定: `p33_eval_v5.py judge`。結果はまずリーダーに報告し、commit しない。
