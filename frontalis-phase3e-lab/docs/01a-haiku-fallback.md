# 言語取得の追加条件（ユーザー承認による変更）

2026-10-15の再開指示で、OpenCodeが応答しない場合のANTHROPIC_API_KEYによるHaiku 5.5呼出しが明示的に許可された。旧「追加有料APIなし」はこの追加条件について置き換える。キーは環境変数からメモリ上だけで読み、コマンド引数・画面・ファイルへ出さない。

OpenCodeの自然文抽出はMiMoが7/8batch、Nemotronが3/8batchで有効なJSONを取得した。タイムアウト・tool使用・構文失敗を消さない。元の2モデルの取得と評価を維持し、Haiku 5.5を別の補完群として同じ32自然文、同じ24会話、同じ8自然文接続会話で測る。結果をモデル名を入れ替えて補完しない。

Haiku追加群のモデルIDはclaude-haiku-5-5、Messages API、max_tokens 8192、sampling指定なし、外部toolsなし。同じpromptとcase IDで、通信・JSON構文／ID失敗だけ1回再試行する。意味や発言の不一致を理由に再生成しない。元の数値実験・学習則・成功基準は変更しない。

モデルIDとsampling仕様の確認：[公式Haiku 5.5仕様](https://platform.claude.com/docs/en/models/haiku-5-5/overview)。
