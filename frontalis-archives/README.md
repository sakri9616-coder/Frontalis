# frontalis 専用アーカイブ

元フォルダごとに独立したZIPとSHA-256マニフェストを保存する。他のPhaseのファイルや稼働中ラボへ混ぜない。各ZIPの最上位フォルダ名は元と同じ。

## 旧ラボ

| 分類 | ZIP |
|---|---|
| 前身core | [nephia-core-lab](labs/core/nephia-core-lab.zip) |
| 前身NPN | [nephia-npn-lab](labs/npn/nephia-npn-lab.zip) |
| Phase 3 | [nephia-phase3-lab](labs/phase-3/nephia-phase3-lab.zip) |
| Phase 3.5 | [nephia-phase35-lab](labs/phase-3.5/nephia-phase35-lab.zip) |
| Phase 3-B | [nephia-phase3b-lab](labs/phase-3b/nephia-phase3b-lab.zip) |
| Phase 3-B自己学習 | [nephia-phase3b-self-lab](labs/phase-3b-self/nephia-phase3b-self-lab.zip) |
| Phase 3-C | [frontalis-phase3c-lab](labs/phase-3c/frontalis-phase3c-lab.zip) |

## 完了した再現コピー

`reproductions/phase-3d/`、`reproductions/phase-3e/`、`reproductions/phase-3f/` に、元フォルダ名を保った個別ZIPを置く。全件と検証状態は [index.json](index.json)。

## 稼働用として保持するもの

3-Dの共有Python/GPU環境と3-E/3-Fの評価画面を使うため、`frontalis-phase3d-lab`、`frontalis-phase3e-lab`、`frontalis-phase3f-lab` は作業ルートの元位置に残す。ASIST本体・実記憶・設定はこのアーカイブに含めない。

## 復元

作業ルートから実行する。既存フォルダには上書き・結合しない。ZIP内全ファイルを検証し、一時展開と内容照合が成功した後に元位置へ戻す。

```powershell
.\frontalis-archives\Restore-Archive.ps1 -Name nephia-phase3b-self-lab
# 旧ラボ7個をまとめて元位置へ戻す場合（展開先が全て未使用であること）
.\frontalis-archives\Restore-Archive.ps1 -Name all-labs
```

元位置を参照する過去の実験・凍結監査をそのまま実行する際は、必要な旧ラボを先に復元する。復元せずに凍結成果の内容一致を調べる場合:

```powershell
& .\frontalis-phase3d-lab\.venv\Scripts\python.exe -B .\frontalis-archives\archive_manager.py audit
```

この監査は元の凍結マニフェストを変更せず、ファイルがアーカイブされた場合はZIP内の同じ相対パスから内容を照合する。結果は [preservation-audit.json](preservation-audit.json)。アーカイブ作成・検証の記録は `archive-progress.txt`。
