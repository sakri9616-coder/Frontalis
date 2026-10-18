# 3-F 学習目標の確認スクリプト

監査（研究思想の理解と現状監査）の第2節の数値を出したスクリプト。3-Fラボの `npnf/` を同じフォルダにコピーして実行する。

```
cp -r <frontalis-phase3f-lab>/npnf audit/
python3 -B audit/probe_learning_target.py audit
```

`shim/torch` はPyTorchが使えない環境向けに、`npnf/core.py` が使う演算だけをNumPyで置き換えた簡易版。元の凍結成果物の再現ではない。PyTorchがある環境では `shim` を外して実行できる。
