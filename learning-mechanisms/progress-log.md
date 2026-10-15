# 進捗ログ（学習機構の検討）

| 時点 | 担当 | 内容 | 成果物 |
| --- | --- | --- | --- |
| 1 | オーケストレーター | 最新push 4fecbad（origin/claude/project-thread-1vq8ai）を取得し、研究思想・監査・再設計・実験計画・試作を読了 | scratchpad/latest/ |
| 2 | オーケストレーター | 課題ブリーフ作成。分担: Sonnet=価値ネット設計/信念ネット設計/批評, Haiku=文献確認/表現検証/記録 | scratchpad/brief.md |
| 3 | Haiku(文献確認) | 参考文献26件の書誌と要旨を検索で照合。確認26 / 訂正0 / 未確認0。詳細のみ未照合5件（#2, #11, #13, #25, #26）は表内に明記 | scratchpad/refs-verified.md |
| 2b | オーケストレーター | 現行則の自己増幅を1次元で解析・数値確認（発散 ½log t、縮小写像による履歴忘却、双安定の反転閾値） | scratchpad/orchestrator-notes.md |
| 4 | Sonnet(信念設計) | 8方式（P0, A階層ベイズ, B潜在原因, C帰属, D後継特徴, E逆計画, F HGF, G自己モデル）を比較。推奨: スコープ帰属つき階層ベイズ信念（SAHB）。toyで持続89%・可塑0.68→0.09、H6aは厳密ベイズでは偏りなし | scratchpad/belief-design.md, belief_sims*.py |
| 5 | Sonnet(価値設計) | 9方式（A現行, A′修復, B TD, C 欲求と設定点, D 学習進捗, E 手段の目的化, F 選択, G 社会, H 次元重み, I ベイズ）を比較。補題1〜4。推奨: アンカー付き多源価値学習（AMV）。最大の弱点: 価値の軸が設計者の所与 | scratchpad/value-design.md, valsim/ |
| 6 | オーケストレーター | 最新pushブランチの作業ブランチへのマージは権限判定で拒否 → 新規文書のみ作業ブランチに追加する方針に変更 | – |
| 7 | オーケストレーター | 統合草稿「配分型フレーム」を作成（教師は外生、人格依存は配分・帰属・信頼・欲求・解釈・標本・学習率の7経路、信念はSAHB＋事前ヘッド、想起の自己増幅を禁止、設定点の更新を関与度で制限） | docs/04-learning-mechanisms.md |
| 8 | Haiku(雑務・出典照合) | 付録 docs/04a・04b を作成（scratchpad 参照をリポジトリ内パスに置換）。玩具 sim0〜5・belief・check_n1 を learning-mechanisms/toys/ に移設し import を相対化。value の out0〜5 は再実行で一致（out3 は sim3 と sim3b の出力）。README 作成。04 の出典27件を refs-verified と WebSearch で照合し、値の不一致なし（Keramati 2014、McClelland 1995 は検索で確認） | docs/04a-value-net-candidates.md, docs/04b-belief-net-candidates.md, learning-mechanisms/README.md |
| 9 | Sonnet(批評) | 草稿を査読。致命2（配分の正帰還 F1、新概念の飢餓 F2）、重大9（û の不整合、設定点は縮小、信頼の裏口、想起の水増し、具体例の符号、研究思想の後退、判定の穴、SAHB の識別性、最有望の根拠） | scratchpad/critique.md, critique/ |
| 10 | オーケストレーター | v2 仕様を作成（カルマン配分、û 廃止、成長欲求の効用、認識的信頼、宣言したコミットメント、1段EC、後悔、retract-and-reassign）。本文を v2 に全面改稿（数値は検証待ち） | scratchpad/v2-spec.md, docs/04 |
| 11 | Sonnet(検証) | v2 を玩具で検証。T1 合格（コミットメント以外の半径 ≤1.000001）、T2 MF 合格・MB 経由のブロッキング残存、T3 持続と可塑性の両立不成立、T4 NV3 は係数依存、T5 閉形式一致、T6 条件付き、T7 K の符号出ず。新欠陥 N1〜N6 | scratchpad/v2-verify.md, critique/v2/ |
| 12 | オーケストレーター | v3 仕様: 価値も個体／類／一般の階層に（持続は未見の個体、可塑性は新しい個体自身で測る）、MB は状況単位、社会は源ごとに正規化、q の学習則、コミットメントはコア外 | scratchpad/v3-spec.md |
| 14 | Sonnet(検証) | v3 を玩具で検証。V1 持続 0.91〜0.93・可塑性 1.00 で合格（v2 0.26/0.74、固定率 0.09/0.90、1/n 0.50/0.50 は不合格）。V2 学習の向きが R −0.065／K +0.618／X −0.067 と分岐（K は A を嫌わない）。V3 半径 1.0000（q 固定）。V4 条件付き合格。V5 NV3 不合格。V6 単独迎合者は合格、協調迎合者で不合格。新欠陥 M1〜M8 | scratchpad/v3-verify.md, critique/v3/ |
| 15 | オーケストレーター | 本文を v3（精度配分・階層型）で確定。検証表、具体例（サブタイプ化の予測）、ループ台帳、未解決（類の抽出、単一ユーザー、NV3）を更新 | docs/04-learning-mechanisms.md |
| 13 | Haiku(雑務) | 付録 docs/04c-critique-and-verification.md を作成（第1部 初稿の批評、第2部 v2 検証、第3部 v3 検証。見出しは一段階下げ、参照は repo 内パスに置換。字句と数値は行単位で照合し不変）。specs/ に v2-spec・v3-spec を複写（同一）。toys/critique/ に c*・v2/・v3/ の .py 45本と .txt を複写（v3/old_q0first/ は下位ディレクトリのため未複写）。28本で import と参照パスを __file__ 基準に変更（数値・ロジックは不変、原本との出力一致を確認）。python3 -I -B で全45本を timeout 300 s で実行: 45本とも exit 0、最長 c1_nullspace 58 s、stderr なし。参照 out との比較: 25本一致。不一致2本は t1_jacobian と t1b_closed で、差は probe ratio の値のみ（他の値は一致）。参照 out（16:21・16:22）が v2model.py の probe() 修正（16:23:45、v2-verify の事後調整 (1)）より前に生成されたためと思われる。参照 out は未変更。README に specs/ と toys/critique/ の節を追加（既存内容は保持） | docs/04c-critique-and-verification.md, learning-mechanisms/specs/, learning-mechanisms/toys/critique/, learning-mechanisms/README.md |
| 16 | Haiku(表現検証) | docs/04 最終版を点検（編集なし）。§3.6・§6・§7・§8 の数値を 04c 第2〜3部・04a・04b・orchestrator-notes と照合。一致: 持続・可塑性表、G4′（2.45・1.030・1.214・0.19倍）、音楽（0.300・0.094・0.364）、迎合者（98/100・0/100・78/100）、§6 の R/K/X 値、類の反転 7%/72%、類の誤り 19%/49%、Weber & Crocker 1983（WebSearch、JPSP 45(5), 961–977）。不一致: §6 閾値の符号（正しくは Σe < +0.12）、§1・§1.5 の ρ≤1.0000（q 学習で 1.214）、§3.6 の事前分散倍率（1/4〜4倍）、§3.2 S4 の 0.987（100回再生）。表の列崩れ: 04 L206・207・213、04c L590。ほか記号の二重定義・未定義（3-F、NPN、a_min、g、ρ、m、k など）と日本語の細部は報告どおり | – |
| 17 | オーケストレーター | 表現検証の指摘を反映: §6 の閾値の符号を訂正（Σe < +0.12）、半径の限定条件、再生回数、表内の `|` のエスケープ、3-F の定義、記号表の追加、未定義の文言。04c には編者注 | docs/04, docs/04c |
