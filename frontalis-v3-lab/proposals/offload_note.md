# offload_note: 別の計算機へ処理を退避する場合の注記（提供元に依らない）

状態: 注記（2026-10-17）。**有料の計算資源は使わない**（ユーザーの方針。特定のクラウドの手順書としての版は取り下げた。旧名 `saturn_offload.md`）。この注記は、無料で使える計算機（手元の別の機械、研究室の共有機など）が手に入った場合に、手元と同じ規則で使うためのもの。計算量そのものを減らす計画は `DESIGN.md` 第 14 節。
記号: **要確認** = 確かめていない手順・値。**要測定** = 最初に測る値。この注記のために、評価用の種と有料の API は使っていない。

---

## 1. 退避に向く作業・向かない作業

| 区分 | 作業 | 理由・注意 |
| --- | --- | --- |
| 向く | 開発用の種での本番規模の確認（`p15_prod.py cells ...`） | 区画ごとのチェックポイントから自動で再開する |
| 向く | リハーサル（`p17_rehearsal.py <学習器> lr`）、κ の確認、規模の曲線、学習率の選び直し | 開発用・リハーサル用の種だけ。判定の量を使わない調整 |
| 向く | VRAM の制約で分けていた区画（例: NN-REL） | VRAM の大きい GPU なら分けずに済む見込み（第 6 節）。分け方で結果が変わらないかは P23 の batch2 の検査で見る |
| 注意 | **評価用の種 77100〜77139** | 1 回限り。**評価の本番は 1 か所でだけ実行する**（区画を機械の間で分けない）。別の機械で行うなら、その前に、その機械で開発用の種を通し、第 4 節の検査に通ること |
| 向かない | bf16 の使用 | 草案 3.1 第 10b 節の検査に不合格（P12）。fp32 のまま |
| 向かない | 認証情報を要する作業の自動化 | 認証情報はユーザーが設定する。第 5 節 |

## 2. セットアップ

手元の環境（2026-10-17 に確認）: Windows 11、Python 3.12.13、torch 2.7.0+cu128（CUDA 12.8、cuDNN 9.7.1）、numpy 2.5.4、GPU は NVIDIA GeForce RTX 3050（VRAM 8 GB、他のエージェントと共有）、ドライバ 617.42、ホストの RAM 32 GB（他の処理と共有）。コードの依存は **torch と numpy だけ**。

```bash
git clone --branch claude/value-representation-design --single-branch https://github.com/sakri9616-coder/Frontalis.git
cd Frontalis/frontalis-v3-lab/proposals/probes
python3.12 -m venv ~/v4d_venv && source ~/v4d_venv/bin/activate       # 要確認: python3.12 の有無
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128   # 要確認: その GPU とドライバへの対応
pip install numpy==2.5.4
nvidia-smi
python -c "import sys, torch, numpy; print(sys.version, torch.__version__, torch.version.cuda, numpy.__version__, torch.cuda.get_device_name(0))"
python -B p13_ckpt_early.py        # 決定性と再開のビット一致: set と krr で identical: True
```

手元との差（OS、Python、torch・CUDA・cuDNN、numpy、GPU と VRAM、ホストの RAM、`p13_ckpt_early.py` の結果）を記録する。`p21_cloud_equiv.py` は環境を結果に書く。numpy の乱数の列が OS に依らず同じかは要確認（違えば第 4 節の検査で分かる）。

## 3. 実行のしかた（中断に強く）

- チェックポイントは Python の `tempfile.gettempdir()` の下に書く。**起動の前に `TMPDIR` を、停止で消えない場所に向ける**。チェックポイントは 1 個で最大約 0.9 GB（240 人格）。
- 待ち行列 `queue_cloud.sh`（`queue4.sh` と同じ形。インタプリタは `$PY`、終わった行は `queue_cloud.done` に書き、再起動の後は飛ばす）:

```bash
export TMPDIR=$HOME/v4d_tmp && mkdir -p $TMPDIR
cp queue_cloud.example.txt queue_cloud.txt          # 中身を編集する
PY=$HOME/v4d_venv/bin/python nohup bash queue_cloud.sh queue_cloud.txt > /dev/null 2>&1 &
tail -f queue_cloud.log
```

- 再開: 同じ `TMPDIR` で同じ待ち行列をもう一度起動する。途中の区画（`p15_prod.py cells`、`p17_rehearsal.py`）は、`cells.run_cell` が構成のハッシュつきのチェックポイントから再開する（ビット一致は P13・`p13_ckpt_early.json` で確認済み）。チェックポイントを持たない作業は最初からやり直す。
- 1 つの GPU で同時に走らせるのは 1 本だけ（手元では、複数を同時に走らせると VRAM が足りなくなった）。

## 4. 手元と別の機械の結果の同等性（結果を見る前に定義した。`p21_cloud_equiv.py`）

浮動小数点の演算の順序が GPU ごとに違うので、ビット一致は期待しない。P18 で、ミニバッチの順だけを変えた再試行が評価の距離の大部分を作ることが分かっているので、小さな数値の違いも SGD の経路に沿って再試行の大きさまで育つ見込みが高い。

- 実行: リハーサルの種 77340〜77342、NN-SET、I-H、C-off・C-SLp、F1・F6m・F5、役 6、2,000 経験、学習率 1e-3。手元の基準は `p21_local_rtx3050.json`（commit 済み）。別の機械で `run other`、`run other_retest retest`、`compare local_rtx3050 other other_retest`。
- 段 1（数値として同等）: E1 δ の中央値 ≤ 0.2 × ν の中央値、E2 δ の 90 百分位 ≤ 0.5 × ν の中央値、E3 非有限 0、E4 `p13_ckpt_early.py` が identical。
- 段 2（再試行として同等）: E5 δ の中央値 ≤ 1.5 × ν の中央値、かつ E3・E4。
- 使い方: 段 1 に通れば、開発用の種の区画を混ぜてよい。段 2 だけなら、開発の証拠として使ってよいが、同じ判定の中で手元の区画と混ぜない（`cells.run_cell` は `meta` に GPU の名前・torch の版・ホスト名を書く）。評価の本番を全区画その機械で行うことは許す。段 2 にも通らなければ、その機械の結果は判断に使わない。
- 手元の自己検査（`p21_compare_selftest_local.json`）: 手元の 2 回の実行は δ = 0 で、段 1・段 2 に通った（比較の計算が動くことの確認だけ）。

## 5. 結果の持ち帰り

- 別の機械の結果は**別のブランチ**に置き、手元の結果を上書きしない（例: `claude/value-representation-design-other-<GPU>-<日付>`）。差は `git diff` で見る。
- 持ち帰りの方法（commit して push するか、ファイルを手で運ぶか）はユーザーが決める。**認証情報はユーザーが設定する。鍵・トークン・パスワードを commit しない**。commit の前に `git status` と `git diff --cached --stat` を見る。チェックポイント（`.pt`）はリポジトリに入れない。
- 評価用の種の結果が出た場合は、push の前にリーダーへ先に報告する。

## 6. メモリの見込み（VRAM とホストの RAM を分ける）

- **GPU の VRAM**（P20、`p20_mem.json`、本番の構成、学習の段だけ）: NN-SET 約 2.1 MiB／人格（＋固定 約 130 MiB）、NN-REL 約 9.1、KRR 約 7.5、RF 約 4.2、HB 約 5.3 MiB／人格。測定の段の最大値は要測定。VRAM の 8 割を学習に使えるとした粗い上限: 16 GB で NN-SET 約 6,000・NN-REL 約 1,400・KRR 約 1,700 人格、24 GB でその 1.5 倍、80 GB で約 5 倍。手元の 8 GB（他のエージェントが約 5.4 GB を使うとき、残り約 2.5 GB）では NN-REL は約 250 人格まで。
- **ホストの RAM**: 手元の失敗はこちら（本番規模 720 人格で、乱数の列 11.0 GiB の `np.stack` が確保できなかった。今のコードは 1,000 経験ごとに作るので、240 人格で区画ひとつ約 123 MB）。チェックポイントを書くときに 1 個ぶん（240 人格で約 0.9 GB）をホストで組み立てる。測定の段のホストの RAM の最大値は要測定。ホストの RAM が 16 GB 以下の機械は避ける。
- 最初に測る項目: `p13_ckpt_early.py`、`p20_mem.py`、`p21_cloud_equiv.py` の 3 行、1 区画の本番規模の最初のチェックポイントまでの時間、その間の VRAM とホストの RAM の最大値。

## 7. GPU 時間

手元の RTX 3050 での GPU 時間の見積もりは `DESIGN.md` 第 14 節（段階ごと）。別の機械での速さは要測定（この処理は小さな行列積とカーネルの起動が律速で、P11 では演算の効率が約 0.4 TFLOP/s だったので、GPU の公表の演算性能の比ほどは速くならない見込み）。
