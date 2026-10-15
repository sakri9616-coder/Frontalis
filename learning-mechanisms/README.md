# learning-mechanisms

docs/04（学習機構）と付録 docs/04a・04b を支える数値スケッチです。実験ではありません。パラメータはすべて手で置いた値で、実データは使っていません。出力は機構の向きと大きさの目安としてだけ読んでください。

## ファイル
- `toys/value/sim0_existing.py` … 現行ルール（prototype）の1次元簡約と長期履歴（出力 out0.txt）
- `toys/value/sim1_candidates.py` … 候補 A, B, C, E, I の3人格×NEW、保持・反転・無関係試験（out1.txt）
- `toys/value/sim2_custom.py` … 候補 B〜I の1次元解析と3人格デモ（out2.txt）
- `toys/value/sim3_defs.py` … v*(i,d) 表、参照モデルとの R²、信念入れ替え分解（out3.txt）
- `toys/value/sim3b_niche.py` … 領域選択（学習進捗ニッチ）の挙動（out3.txt の末尾）
- `toys/value/sim4_repaired.py` … 候補 A′（out4.txt）
- `toys/value/sim5_amv.py` … 統合玩具 AMV-L1, L2（out5.txt）
- `toys/value/prototype.py` … sim0〜sim5 が import する試作コードの写し
- `toys/belief/belief_sims.py`, `belief_sims_extra.py` … 信念ネット候補の玩具検証（04b）
- `toys/check_n1.py` … オーケストレーターの解析メモ N1, N2, N4 の数値確認
- `orchestrator-notes.md`, `refs-verified.md` … 解析メモ（N1〜N6）と文献確認表

## 実行
リポジトリのルートで次のように実行します（スクリプトは自分の場所を基準に import するので、作業ディレクトリは問いません）。
```
python3 -I -B learning-mechanisms/toys/value/sim0_existing.py
```
必要なのは Python 3 と numpy だけです。`-I` は環境変数の影響を避ける指定、`-B` は `__pycache__` を作らない指定です。
