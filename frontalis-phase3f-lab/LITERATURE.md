# 実測知見と3-Fの設計仮説

2026-10-15に下記の原著論文の本文または著者抄録を確認した。人間の心理をこのNPNと同一視しない。感情語を7つの普遍的実体と仮定しない。

| 領域 | 確認した原著の知見と限界 | この実装で試す独自仮説 |
|---|---|---|
| 持続・反すう | Verduyn & Lavrijsen (2015), *Motivation and Emotion* 39,119–127。233名の高校生の回顧報告で、感情の持続と出来事の重要性・反すうの関連を検討。持続の定義や回顧バイアスの制約があり、普遍的な減衰定数は得られない。[著者公開本文](https://ppw.kuleuven.be/okp/_pdf/Verduyn2015WELLA.pdf), DOI 10.1007/s11031-014-9445-y。 | 再活性化は同じ当時の解釈への新しい情報更新として扱い、短期・気分・係数の各持続を測る。0.45/0.9という残存率は独自設定。 |
| 感情慣性 | Kuppens, Allen & Sheeber (2010), *Psychological Science* 21,984–991。経験サンプリングと親子相互作用の時系列で自己相関として慣性を評価。関連研究であり、高慣性をNPNの不良人格と判定する根拠ではない。[本文](https://pmc.ncbi.nlm.nih.gov/articles/PMC2901421/), DOI 10.1177/0956797610372634。 | 感情慣性を更新遅れの観測量として測る。無情報区間の決定論的減衰曲線の自己相関は定数を反映するため、人間研究と同じ証拠とは扱わない。 |
| 文脈による交絡 | Koval, Sütterlin & Kuppens (2016), *Frontiers in Psychology*。感情文脈を統制した慣性とwell-beingの関連を検討。[本文](https://pmc.ncbi.nlm.nih.gov/articles/PMC4705270/), DOI 10.3389/fpsyg.2015.01997。 | 同じ現在を4つの過去条件で比較し、入力の差による反応と内部状態の持続を分ける。人格の良し悪しの尺度へ流用しない。 |
| 消去と再発 | Urcelay et al. (2024), *Renewal of instrumental avoidance in humans*。2実験のABA/ABB比較で、消去した回避の元文脈における回復を観測。[原著抄録](https://pubmed.ncbi.nlm.nih.gov/39101917/), DOI 10.1037/xan0000383。人間の回避課題であり一般的な謝罪受容の実験ではない。 | 新しい文脈での反対結果を追記し、元の文脈の評価がどの程度残るか検証する。係数を単純に0へ戻す忘却と新学習を別操作とする。RBF局所性による文脈保持は生物学的消去の再現とは主張しない。 |
| 時間と消去 | Huff et al. (2009), *Delayed extinction attenuates conditioned fear renewal and spontaneous recovery in humans*。5分/1日と同/異文脈を操作して回復差を検証。[原著抄録](https://pubmed.ncbi.nlm.nih.gov/19634943/), DOI 10.1037/a0016511。 | 経験の順番と空白更新数を独立操作する。人間の1日を更新何回と換算する根拠にはしない。 |
| 感情調整 | Gross (1998), *JPSP* 74,224–237。120名の映像課題で再評価と表出抑制を比較し、経験・表情・生理反応の差を報告。[原著抄録](https://pubmed.ncbi.nlm.nih.gov/9457784/), DOI 10.1037/0022-3514.74.1.224。 | 内部感情とLLMの発言を別計測。数値状態を0.5倍する介入は操作的感度分析で、再評価や表出抑制の生理モデルではない。訂正された解釈を新経験として与える条件も別に設ける。 |
| 多言語 | Jackson et al. (2019), *Science* 366,1517–1522。2,474言語の共語化で感情概念の文化差と快不快・活性の共有構造を検討。語彙の構造から各個人の感情体験を直接測定した研究ではない。[著者公開本文](https://static1.squarespace.com/static/5d8c8bd71a675f210c9996e6/t/5e0266a99bb1cd38ee8a0611/1577215657923/Jackson%2BScience%2B2019.pdf), DOI 10.1126/science.aaw8160。 | 内部を言語に依存しない連続数値7軸とし、喜び方向と悲しみ方向の正値の同時活性も許す。互いに排他的な感情ラベルや翻訳辞書を内部状態にしない。7軸とその投影は3-E由来の操作的設計で、文献が7軸を実証したわけではない。日英仏西の同一信号の発言差を調べる。 |

## ASIST日記の読み取り調査

実装ソースと雛形のみを調査し、userDataの実日記、me.md、user.md、設定は読まず、変更しない。
`resources/skills/memory-templates/ja-JP/journal.md`は日付見出し、話題ごとの一人称記録、最後の「今日の私」。`src/shared/ipc.ts:683`のMemoryUnitはid/file/line/kind/page/heading/aliases/text/date/order、`memory-page.ts:156`は日付と見出しを検索単位へ変換する。`memory-store.ts:46`のjournalディレクトリ、curation処理、`brain/memory-tools.ts`の説明から、日記は現在の確定事実ではなく当時の経緯だと確認した。ADR0052は11言語の雛形と過去言語との互換を説明する。

研究用の対応は、date→情報更新番号、topic→経験ID、当時の一人称の見方→interpretation、根拠のある出来事→facts、振返り→当時のNPN評価、関係するページ→時間的links。ASISTのMarkdownをNPNへ直接投入せず、架空の構造化JSONとして実装する。追記専用・訂正参照・確信度・7数値軸はこの研究で追加した構造であり、ASISTに既存のNPN機能があるとは言わない。
