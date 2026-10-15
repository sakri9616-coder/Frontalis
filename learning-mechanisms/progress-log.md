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
| 18 | ユーザー | 要求: 理論設計書ではなく実際に動作する NPN の最小実装（価値ネット・信念ネット・学習可能な NN の統合、経験→評価→学習→次の評価、履歴差・継続学習・保存復元の検証）。批評は理念優先。ここまでの費用は約 $30 | – |
| 19 | オーケストレーター | 環境確認: Python 3.13, numpy 2.5.3 のみ（scipy・pytest・torch なし）→ numpy のみ・unittest で実装する方針。実装仕様と事前固定の判定基準を作成 | frontalis-v3-lab/SPEC.md |
| 20 | Sonnet(実装) | v3.0 最小実装（npnv3、numpy のみ、57 テスト約2分）。T2(c) のみ不合格、T1・T9・T4 に事後変更（閾値は不変）。REPORT.md はツール制限でサブエージェントから保存不可 | frontalis-v3-lab/ |
| 21 | Sonnet(理念批評) | v3.0 を理念優先で批評。理念違反3（受け止め方が学習状態に依存しない／評価基準が形成されず M の写し／循環が閉じていない）、重大5（NN 未使用、種20本が実質1標本、T2(a) は状態を測定、事後変更が結論を支える、埋め込みが LLM に束縛）。判定: NPN の最小フレームとは呼べない | scratchpad/critique-impl.md |
| 22 | オーケストレーター | v3.1 仕様: 欲求の関心の重み a、随伴による条件づけ価値 C（TD, γ<1）、受け止め方 f に学習状態と予期を入れる、曖昧な結果の信念による解釈、NN の混合と基線比較、埋め込みを NPN が所有、κ の経験ベイズ、確率的な経験史。判定基準 U1〜U9 を実装前に固定 | frontalis-v3-lab/SPEC-v3.1.md |
| 23 | Sonnet(実装) | v3.1 実装（102 テスト約3.5分）。U1〜U7・U9 合格、U8（全体の Q の持続 0.564/0.352）不合格。T1〜T10 維持（T2(c) は expectedFailure）。NN は 1-NN に勝たないことを正直に報告 | frontalis-v3-lab/ |
| 24 | Sonnet(実装) | 改訂1（類×結果型ごとの κ、Kemp et al. 2007）。U8 は 0.559/0.347 で不合格のまま。再診断: 主因は κ ではなく、責任 φ を経験時に凍結すること（文脈の層が N の経験で上書きされる） | CHANGELOG G 節 |
| 25 | オーケストレーター | 改訂2を承認: 想起による帰属のやり直し（retract-and-reassign、オンライン EM、docs/04 第5節の手順7） | – |
| 26 | Sonnet(理念批評) | v3.1 再批評。判定「限定つきで可」。理念違反 P1（a・C → 学習の経路がない）、P2（a は曝露統計で方向は仮説）。重大: C は行き止まり、U1 の識別力、世界表が対比を作る、κ 改訂の前提は否定、NN は 1-NN に勝たない | reviews/critique-v3.1.md |
| 27 | Sonnet(実装) | 改訂2（想起・再帰属、オンラインEM、K_r=10）。信念の持続は m=32 で 0.54→0.71（同じ場面）、0.60→0.91（未見の場面）。Q の U8 は 0.567/0.411 で不合格。残る原因は場面の価値の可塑性と欲求状態への依存（設計どおり）。π は固定点に収束し、張り付きは単一説明の長い履歴でのみ | CHANGELOG G' 節 |
| 28 | オーケストレーター | v3.2 実装を指示。U8 は不合格のまま残し、診断に基づく U8′（状態を揃え、未見の場面で、人に帰属する部分の持続）を事後定義として追加 | – |
| 29 | Haiku(再現確認) | 新規 clone（claude/charming-galileo-wawr7u、HEAD acc237d835512dc6bb4993a9df6c8e33b3d4c75e）で再現。unittest 146 件、不合格 4（U8、U8′可塑性 m=8、W2、W4(a)）、expectedFailure 3、所要 351 秒（REPORT の約 330 秒より長い）。demo は exit 0・約 2.3 秒。CLI の init・feed・show は exit 0。REPORT §8 の診断と経験史の再生成も exit 0、tracked ファイルの差分なし。REPORT §9 は未記入 | /tmp/claude-0/-home-user-Frontalis/72635c51-dda1-5f12-9bae-7b9ed27bb621/scratchpad/repro/test_run.log |
