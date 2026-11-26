# CLAUDE.md

このリポジトリは 2026-11-26 から Claude（Claude Code）が開発を引き継いでいます。

作業を始める前に必ず次の 2 つを読んでください（内容はここに重複させません）。

1. [GEMINI.md](./GEMINI.md): プロジェクト憲法・設計不変条件・コードベース索引（Antigravity / Gemini 時代から共通）
2. [docs/HANDOVER_CLAUDE.md](./docs/HANDOVER_CLAUDE.md): 引継ぎの経緯、中断時点の状態、使い方、既知の問題

Claude 固有の運用メモ:

- 検証コマンド: `.venv\Scripts\ruff.exe check --select E,F,W,I --ignore E501 src tests` と `.venv\Scripts\python.exe -m pytest tests -q`
  - フル設定の `ruff check src tests` は既存の指摘が約 129 件あるので、変更で増やさないことを基準にする
- レンダーは GPU と実機 VST3 を使うため、`mix-auto mix` を同時に複数走らせない（1 曲ずつ順番に）
- Studio One で開いている出力フォルダは Media がロックされる。パイプラインは自動で `<曲>_auto_YYYYMMDD-HHMMSS` に書き出す
- コミットには Claude が行った作業だと分かるメッセージを付ける
