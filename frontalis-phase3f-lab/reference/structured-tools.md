# 追加比較 — 型を固定した抽出と実際のNPNツール呼出し

主抽出契約は8軸の順を説明する一方、配列の実例が不足していた。Haikuの欠測には軸名付きオブジェクトが含まれ、これをすべて意味理解の失敗と扱えない。主16件の厳密結果は維持し、同じ自然文・意味方針・2件バッチにJSON Schemaだけを追加したHaiku条件を別実施する。値を補完・人手訂正せず、型の失敗と意味内容の誤差を分ける。
公式仕様（2026-10-15確認）: [Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs), [Strict tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use)。スキーマ保証を事実の正確さの保証と取り違えない。

さらに5容量×{初期、訂正後}=10件で、同じgold意味をLLMに与えてevaluate_npnを呼ばせる。実際の入力をstrict dispatcherで検証して保存人格を評価し、tool_resultをLLMへ返す。意味は既に与えるため、ここは意味抽出の採点ではない。元の要求との一致、正しいモデルと意味への問合せ、読取前後の不変、最終発言の選択・態度一致を個別計測。ツールはread-only、学習・ファイルパス指定は提供しない。人格名・初期傾向の説明を与えない。1回のツール往復で止め、追加要求はそのまま失敗として保存する。人間評価とは別の追加条件である。
