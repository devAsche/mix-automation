# MIX Automation (`C:\MIX_automation`) — Antigravity プロジェクト憲法・開発ガイド

本ファイルは Antigravity のプロジェクト **MIX_automation** で開かれるすべての新規会話に自動ロードされる中核コンテキストである。詳細な履歴・数値・仕様は `docs/` および `.agents/skills/` に分離しているため、必要に応じて該当ファイルを参照すること。

## 1. プロジェクト概要と実行環境

- **目的**: 歌ってみた制作において「どの楽曲・声域・編成でも破綻せずオケの内側へ溶け込む商業・一流MIX師クオリティ」と「Studio One 5.5 での 1:1 再現・圧倒的時短・クリエイティブ演出／ディレクション支援」を統合実現する自動ボーカルMIXエンジン。
- **GitHub リポジトリ**: `https://github.com/devAsche/mix-automation`（ブランチ `main`、現在バージョン `v6.0` / `0.6.0`）
- **Python 仮想環境**: `C:\MIX_automation\.venv`（Python 3.12 / `uv` 管理、CLI: `C:\MIX_automation\.venv\Scripts\mix-auto.exe`）
- **ローカル実機**: Windows / NVIDIA GeForce RTX 5060 Ti (16GB VRAM) / PreSonus Studio One `5.5.2.86528`（`.dawproject` 非対応のためネイティブ `.song` ZIP + XML + Steinberg `.vstpreset` 注入方式を採用）
- **オーディオ・成果物ディレクトリ**（`.gitignore` 対象）:
  - 入力素材: `data/raw/<曲名>/`（サブフォルダ型 `stems/`・フラット配置の両方に対応）
  - リファレンス音源: `data/references/`（8歌い手・44曲＋公式オフボペア）
  - ベーステンプレート: `data/templates/test20261124/test20261124.song`
  - 出力先: `data/outputs/<曲名>_<style>/`（マスターWAV、ボーカルバスWAV、`.song`、`mix_report.json`）

## 2. ユーザーの 4 段階ロードマップ（現在地: v6.0）

1. **Stage ①（約 88%）**: 複数曲で破綻しないオケと声の混ざり具合（距離感・張り付き・安定感）の獲得、Studio One 5 での 1:1 数値再現、1パートの手直しを全パートへ伝播するアンカー伝播。
2. **Stage ②（約 78%）**: ノイズ処理・ピッチ／タイミング補正・音均し・EQ・コンプ・質感調整をボタン 1 つで一貫処理し、オケの音圧変化に対して日英歌詞単位で芯と距離感を安定維持。
3. **Stage ③（約 42%）**: 儚く抜けるリバーブスロー、オートチューン、ラジオボイス等の本家スタイル特殊効果を歌詞・秒単位で検出・再現。
4. **Stage ④（約 52%）**: コーラス旋律・ハモリの提案と仮歌ガイド生成など、歌ってみたディレクターとしての協働制作。

> 原文・達成状況の詳細・未解決課題・次期 R&D 方針は `docs/ROADMAP.md` を参照。

## 3. 絶対に破ってはならない設計原則（不変条件）

1. **個別曲の対症療法を禁止し、常に構造的汎用化を行う**: 1曲の聴感指摘（声浮き、歯擦音、静寂パートの違和感など）を固定マジックナンバーで誤魔化さず、物理・心理音響・アーキテクチャの原因を特定して全曲へ効く仕組みに昇華する。
2. **1 から作る前に GitHub の優良 OSS を徹底調査する**: すでに `spotify/pedalboard`、`nomadkaraoke/python-audio-separator` (`BS-RoFormer`)、`csteinmetz1/dasp-pytorch`、`SonyResearch/diffvox`、`legsmechanical/bark-transient-shaper` 等の知見を統合済み。新機能着手前に必ず既存 OSS の流用・昇華を検討する。
3. **決定論的 DSP 最適化とクリエイティブ判断を分離する**: 0.1 dB / ms 単位の EQ・コンプ・レゾナンス制御は数理最適化で決定論的に計算し、LLM 的判断は演出・ハモリ提案・重み調整に限定する。
4. **Studio One 5.5 へ書き出す `.vstpreset` は必ずネイティブ Steinberg `Comp`/`Cont` へ展開する**: Pedalboard の JUCE `<VST3PluginState>` ラッパーをそのまま埋め込むと Studio One がノブ設定を無視して初期値にフォールバックする（JUCE リーク 0 件を常時維持すること）。
5. **Melodyne 済みテイクに対してデフォルトで破壊的なピッチ・タイミング補正を行わない**: `--auto-pitch` / `--auto-align` は明示オプトインとし、通常は Studio One レビューマーカー出力にとどめ、グルーヴやリード／ハモリ同期を守る。
6. **高域ジリジリ・暗騒音持ち上げの過去バグを再発させない**: `Waves MV2` 過剰アップワード圧縮、遅い 1 段目 VCA アタック（`attack=1.75` + `drive=1.0`）、全帯域サチュレーション（現在は `240 Hz–4.0 kHz` 中域限定バンドパス）、アンマスキング二重適用を絶対に再導入しない。
7. **検証と GitHub 記録の徹底**: 変更後は必ず `ruff` と `pytest`（全 21 テスト：単体 15＋実 VST3 契約 6）を実行し、重要な開発マイルストーンはドキュメント更新と併せて `devAsche/mix-automation` へ記録・プッシュする。破壊的コマンドは必ず事前確認を取る。

## 4. コードベース構成

- `src/mix_automation/cli.py`: Typer CLI（`mix`, `build-references`, `propagate-anchor`, `doctor`, `inspect-plugin`, `analyze-pair`, `inspect-song`）
- `src/mix_automation/pipeline.py`: v6 エンドツーエンド統合パイプライン（入力自動検出 → 下処理 → ターゲット合成 → 距離感トラッカー → 最適化 → ハモリ／コーラス分離 → Stage ③/④ → マスタリング → Studio One `.song` 出力）
- `src/mix_automation/analysis/relative_profile.py`: 曲非依存の対オケ相対音響プロファイル（K-LUFS、Mid K-LUFS、部分ラウドネス、7 帯域固有声質・対オケマージン、Wet-Tail、Side/Mid、アーキタイプ分類、アンカー伝播）
- `src/mix_automation/analysis/reference_extractor.py`: `BS-RoFormer` 音源分離、44 曲公式オフボ較正、9-NN 連続音響ターゲット合成＋テイストブレンド（既定 `--taste azusagawa --taste-weight 0.25`）
- `src/mix_automation/config/plugin_registry.py`: 実機検証済み 27 VST3 プラグインレジストリ（`VocAlign 6` 期限切れ除外、`iZotope VocalSynth 2` ライセンス済み統合）
- `src/mix_automation/dsp/preprocess.py`: `RX 10` 下処理、音節ラウドネス均し、6 帯域 Bark 子音トランジェントソフトシェイパー
- `src/mix_automation/dsp/pitch_timing.py`: 非破壊ピッチ／タイミング解析、レビューマーカー生成、オプトイン補正
- `src/mix_automation/dsp/vst3_adapter.py`: 厳格な VST3 パラメータ型強制・書き戻し検証アダプタ（`Soothe2` kHz トークン、`Persistent Q` 16 バンド、`LA-2A` ユニティ較正、`VocalSynth 2` 等）
- `src/mix_automation/dsp/vst3_optimizer.py`: 10 帯域 PyTorch 微分彫刻 EQ（`z_db.detach()` 中低域保護）、自動狭 Q ノッチ、実 VST3 CMA-ES 最適化、14 帯域 Bark Mid アンマスキング、連続距離感トラッカー、メインハモリ／コーラス独立バス処理
- `src/mix_automation/creative/fx_and_harmony.py`: 本家 FX イベント検出（リバーブスロー、ラジオボイス）、`VocalSynth 2` ラジオ・Biovox 適用、`Nectar 3` ガイドハモリ生成
- `src/mix_automation/io/audio_loader.py`: ステム自動分類（リード・メインハモリ・コーラス・ダブル・オフボ・本家）とモノラル偽ステレオ補正
- `src/mix_automation/io/song_builder.py` / `studio_one.py`: Studio One 5.5 ネイティブ `.song` / `.vstpreset` 生成・検査（JUCE Base64 デコード、`DEFAULT_S1_PLUGIN_SPECS`）
- `tests/test_core.py` & `tests/test_vst3_contracts.py`: 単体・実機 VST3 契約テストスイート

## 5. ドキュメント＆スキル索引（必要時に読むこと）

- `docs/ROADMAP.md` — 4 段階ロードマップ原文・進捗・既知限界・次期 R&D（時間変化 DSP / `muse` 大規模 DB / S1 差分好み学習など）
- `docs/ARCHITECTURE.md` — v6 詳細信号フロー・数理最適化・OSS アセット一覧
- `docs/CHANGELOG_AND_DECISIONS.md` — v1→v6 の全変遷、ユーザー試聴フィードバック履歴、却下した設計とその理由
- `docs/VST3_AND_STUDIO_ONE_HANDBOOK.md` — 27 プラグイン実機仕様・地雷・Studio One ClassID と XML/Preset 構造
- `docs/BENCHMARKS_AND_REFERENCES.md` — 44 曲リファレンス DB 統計と 3 曲ベンチマーク検証値
- `docs/reports/` — インタラクティブ HTML ダッシュボード（`stage4_roadmap.html`, `daw_3songs_console.html`）
- `.agents/skills/automix-workflow/SKILL.md` — MIX 実行・テスト・品質監査手順
- `.agents/skills/vst3-and-studio-one/SKILL.md` — VST3 開発＆Studio One 5.5 書き出し実務ガイド
- `.agents/skills/reference-db-pipeline/SKILL.md` — `muse` 含むリファレンス追加・検品・再構築手順
