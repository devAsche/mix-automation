# CLAUDE.md

このリポジトリは 2026-11-26 から Claude（Claude Code）が開発を引き継いでいます。

作業を始める前に必ず次の 2 つを読んでください（内容はここに重複させません）。

1. [GEMINI.md](./GEMINI.md): プロジェクト憲法・設計不変条件・コードベース索引（Antigravity / Gemini 時代から共通）
2. [docs/HANDOVER_CLAUDE.md](./docs/HANDOVER_CLAUDE.md): 引継ぎの経緯、中断時点の状態、使い方、既知の問題

現在は v7（Gemini 版）から v8 への作り直し中です。方針と段階: [docs/V8_PLAN.md](./docs/V8_PLAN.md)。ユーザー評価（正解データ）: [docs/evaluations/](./docs/evaluations/README.md)。

Claude 固有の運用メモ:

- 検証は `.venv\Scripts\python.exe scripts\dev_check.py`（RESULT: OK を確認）
  - 中身: `ruff check --select E,F,W,I --ignore E501 src tests` が 0 件、フル設定 `ruff check src tests` が基準 129 件から増えていないこと、`pytest tests -q`
  - 既存の指摘を直して件数が減ったら `scripts/dev_check.py` の `FULL_RUFF_BASELINE` を下げる
- レンダーは GPU と実機 VST3 を使うため、`mix-auto mix` を同時に複数走らせない（1 曲ずつ順番に）
- `mix-auto mix … --engine v7|v8`（既定 v7）。`mix_report.json` に engine / git_commit / git_dirty / profiles_sha256 が残る
- Gemini 版（v7）そのものを動かすときは `C:\MIX_automation_gemini`（worktree @ `gemini-final`、専用 `.venv`）で、`-p C:\MIX_automation_gemini\profiles\pro_target_profiles.json -o C:\MIX_automation_gemini\outputs` を付ける（メインの `data\outputs` に書かない）
- Studio One で開いている出力フォルダは Media がロックされる。パイプラインは自動で `<曲>_auto_YYYYMMDD-HHMMSS` に書き出す
- コミットには Claude が行った作業だと分かるメッセージを付ける
- `tools/gdown_yoidore.txt`（cookie のパス入り）はコミットしない（.gitignore 済み）

## 運用ルール（v8 開発）

- **段階（Phase）ごとにセッションを分ける。** 1 セッション = V8_PLAN.md の 1 段階 = 1 ブランチ / PR。前の段階の経緯は PR とドキュメントに残し、会話に頼らない。
- **報告は要点のみ。** ログ・コード全文は貼らない。ruff / pytest は件数と失敗分だけ。数値は短い表で。
- **既存スクリプトを使う。** 検証は `scripts/dev_check.py`、解析は `scripts/analysis/`:
  - `time_to_section.py <曲> 0:20 …` / `--issues docs/evaluations/issues.yaml`: 時刻 → 区間の対応表
  - `section_spectral_diff.py <曲> [--pre raw|prepared|<file>] [--at 0:20 --win 1]`: 区間ごとの処理前後スペクトル差分
  - `section_balance.py <曲> [--at …]`: 区間ごとの V/I 比・Side/Mid・部分ラウドネス余裕
  - `compare_renders.py <A> <B>`: 同じ曲の 2 つのレンダーの LUFS・ヌル残差・mix_report 数値の差（再現確認・A/B）
  - その場で書いた解析で再利用できるものは `scripts/analysis/` に足す（出力は短い表だけ）
- **`.claude/agents` の使い分け:**
  - `implementer`（Opus / effort medium）: 1 段階内の実装とテスト。報告は差分要約と結果表のみ
  - `worker`（Sonnet）: レンダー実行、ログ確認、テスト実行、調べ物の下調べ。コードは変えない
  - 設計・判断・段階をまたぐレビューはメインのセッションで行う
- データの削除・上書きが必要になったら実行せずに報告して止まる。
