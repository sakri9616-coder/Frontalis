# ライセンス（Project frontalis）

このリポジトリの成果物は、種類ごとに次のライセンスで公開します。判断の基準は「何を再利用してほしいか」です。

| 種類 | ライセンス | 全文 |
| --- | --- | --- |
| ソースコード | Apache License 2.0（著作権表示は [NOTICE](NOTICE)） | [LICENSE](LICENSE) |
| 学習済み NN の重み・保存された人格の状態 | Apache License 2.0 | [LICENSE](LICENSE) |
| 研究文書・実験結果 | Creative Commons Attribution 4.0 International (CC BY 4.0) | [LICENSE-CC-BY-4.0](LICENSE-CC-BY-4.0) |
| 学習データ | **未指定。権利関係を確認してから別途指定します** | — |

## 対象の目安（パスの規則）

ブランチや世代によって、フォルダの構成は少し異なります。以下は目安です。迷う場合は、このファイルの種類の欄を優先します。

| 種類 | 主な対象 |
| --- | --- |
| ソースコード（Apache-2.0） | `*.py`、`*.ps1`、`*.sh`、`scripts/`、`tests/` の `.py`、`npnv3/`、`npnf/`、`experiments/`、`proposals/probes/` の `.py` |
| 学習済みの重み・人格の状態（Apache-2.0） | `models/` 配下の `.npz`・`.json` など、保存された NPN の状態 |
| 研究文書（CC BY 4.0） | `*.md`（README、仕様、事前登録、報告、批評、CHANGELOG、設計書）、`docs/`、`reviews/`、`reference/` |
| 実験結果（CC BY 4.0） | `tests/results/`、`reports/`、`proposals/probes/` の結果ファイル（`*.json`、`LOG-dev.md`）など |
| 学習データ（未指定） | `data/` 配下（経験史の `*.jsonl`、辞書など）、LLM が生成した会話ログ |

## 学習データについて

次の点を確認してから、別途ライセンスを指定します。それまでは、**再利用の許可を与えていません**（閲覧のみ）。

- 経験史は生成器が作った合成データですが、乱数だけでなく、設計者が書いた確率表や辞書を含みます。
- LLM（Claude、DeepSeek など）の出力を含むデータは、各サービスの利用規約が再配布に関わる可能性があります。
- 外部の文献・ツール由来の内容を含む場合は、その権利者の条件に従います。

## 第三者の成果物

`frontalis-archives/` などに同梱された第三者のライブラリ・資料には、それぞれのライセンスが適用されます。このファイルの規定は、本プロジェクトが作成した部分に限ります。

## 引用のしかた

CC BY 4.0 の文書・結果を使う場合は、出典として「Project frontalis（Sakuri9616）」と、リポジトリの URL（https://github.com/sakri9616-coder/Frontalis）を示してください。

---

*注意: これは公開時の意図をまとめたものであり、法的な助言ではありません。重要な判断が必要な場合は、専門家に確認してください。*
