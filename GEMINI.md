# MIX Automation (`C:\MIX_automation`) — Antigravity プロジェクト憲法・開発ガイド

> **2026-11-26 引継ぎ**: Gemini の利用制限により、v7.1 以降は Claude（Claude Code）が開発を引き継いだ。Claude 側の入口は `CLAUDE.md`、引継ぎの経緯・中断時点の状態・使い方・既知の問題は `docs/HANDOVER_CLAUDE.md` にある。本ファイルの不変条件は引き続き両者共通のルールとする。

本ファイルは Antigravity のプロジェクト **MIX_automation** で開かれるすべての新規会話に自動ロードされる中核コンテキストである。詳細な履歴・数値・仕様は `docs/` および `.agents/skills/` に分離しているため、必要に応じて該当ファイルを参照すること。

## 1. プロジェクト概要と実行環境

- **目的**: 歌ってみた制作において「どの楽曲・声域・編成でも破綻せずオケの内側へ溶け込む商業・一流MIX師クオリティ」と「Studio One 5.5 での 1:1 再現・圧倒的時短・クリエイティブ演出／ディレクション支援」を統合実現する自動ボーカルMIXエンジン。
- **GitHub リポジトリ**: `https://github.com/devAsche/mix-automation`（ブランチ `main`、現在バージョン `v7.0` / `0.7.0`）
- **Python 仮想環境**: `C:\MIX_automation\.venv`（Python 3.12 / `uv` 管理、CLI: `C:\MIX_automation\.venv\Scripts\mix-auto.exe`）
- **ローカル実機**: Windows / NVIDIA GeForce RTX 5060 Ti (16GB VRAM, CUDA 有効) / PreSonus Studio One `5.5.2.86528`（`.dawproject` 非対応のためネイティブ `.song` ZIP + XML + Steinberg `.vstpreset` 注入方式を採用）
- **オーディオ・成果物ディレクトリ**（`.gitignore` 対象）:
  - 入力素材: `data/raw/<曲名>/`（サブフォルダ型 `stems/`・フラット配置の両方に対応）
  - リファレンス音源: `data/references/`（8歌い手・44曲＋公式オフボペア）および `youtube_references/`（`muse` 新規収集ステージング＆自動検品対象）
  - ベーステンプレート: `data/templates/test20261124/test20261124.song`
  - 出力先: `data/outputs/<曲名>_<style>/`（マスターWAV、ボーカルバスWAV、`.song`、4トラック `.mid` ガイド、`mix_report.json`）

## 2. ユーザーの 4 段階ロードマップ（現在地: v7.0）

1. **Stage ①（約 94%）**: 複数曲（未マスタリング極小音量オフボ含む全 4 曲）で破綻しない混ざり具合、Studio One 5 での動的細分化イベント＋Stage ③ スポット VST3 を含む 1:1 再現、`.song` 手直し差分からの好み学習（`learn-preferences`）と高度アンカー伝播（`propagate-anchor`）。
2. **Stage ②（約 88%）**: オフボ自動ヘッドルームステージング、`RX 10` 下処理、日英歌詞・母音／子音境界アライメントによる子音アタック保護、CUDA GPU 10 帯域微分 EQ＋セクション別 4 帯域ゼロ位相適応彫刻、連続テンポ追跡の一貫処理。
3. **Stage ③（約 78%）**: 本家ボーカル照合＋歌詞フレーズ同期による 5 クラス特殊効果（`ethereal_reverb_throw`, `radio_voice` [`VocalSynth 2`], `hard_autotune` [`MAutoPitch`], `spot_delay_throw` [`kHs Dual Delay`], `distortion_grit` [`Trash 2`]）の検出・2mix 適用・Studio One ライブ VST3 トラック展開。
4. **Stage ④（約 76%）**: `Nectar 3`＋`VocalSynth 2` Biovox のコード整合ハモリ生成に加え、休符へのコール＆レスポンスとラスサビ対旋律（オブリガート）提案・4 トラック標準 MIDI ファイル（`*_Director_Harmony_Guide.mid`）自動書き出し。

> 原文・達成状況の詳細・未解決課題・次期 R&D 方針は `docs/ROADMAP.md` を参照。

## 3. 絶対に破ってはならない設計原則（不変条件）

1. **個別曲の対症療法を禁止し、常に構造的汎用化を行う**: 1曲の聴感指摘（声浮き、歯擦音、未マスタリング極小オフボ、シンコペーションでの倍音テンポ誤検出など）を固定マジックナンバーで誤魔化さず、物理・心理音響・アーキテクチャの原因を特定して全曲へ効く仕組みに昇華する。
2. **1 から作る前に GitHub の優良 OSS を徹底調査する**: `spotify/pedalboard`、`nomadkaraoke/python-audio-separator` (`BS-RoFormer`)、`csteinmetz1/dasp-pytorch`、`legsmechanical/bark-transient-shaper`、`torchaudio` (`MMS_FA`)、`pykakasi`、`mido` 等を統合済み。新機能着手前に必ず既存 OSS の流用・昇華を検討する。
3. **決定論的 DSP 最適化とクリエイティブ判断を分離する**: 0.1 dB / ms 単位の EQ・コンプ・レゾナンス制御は数理最適化で決定論的に計算し、LLM 的判断は演出・ハモリ提案・重み調整に限定する。
4. **Studio One 5.5 へ書き出す `.vstpreset` は必ずネイティブ Steinberg `Comp`/`Cont` へ展開する**: Pedalboard の JUCE `<VST3PluginState>` ラッパーをそのまま埋め込むと Studio One がノブ設定を無視して初期値にフォールバックする（JUCE リーク 0 件を常時維持すること）。
5. **Melodyne 済みテイクに対してデフォルトで破壊的なピッチ・タイミング補正を行わない**: `--auto-pitch` / `--auto-align` は明示オプトインとし、通常は歌詞スニペット付き Studio One レビューマーカー出力にとどめ、グルーヴやリード／ハモリ同期を守る。
6. **高域ジリジリ・暗騒音持ち上げの過去バグを再発させない**: `Waves MV2` 過剰アップワード圧縮、遅い 1 段目 VCA アタック（`attack=1.75` + `drive=1.0`）、全帯域サチュレーション（現在は `240 Hz–4.0 kHz` 中域限定バンドパス、`Trash 2` も `380–4200 Hz` 制限）、アンマスキング二重適用を絶対に再導入しない。
7. **検証と GitHub 記録の徹底**: 変更後は必ず `ruff` と `pytest`（全 25 テスト：単体 18＋実 VST3 契約 7）を実行し、重要な開発マイルストーンはドキュメント更新と併せて `devAsche/mix-automation` へ記録・プッシュする。破壊的コマンドは必ず事前確認を取る。

## 4. コードベース構成

- `src/mix_automation/cli.py`: Typer CLI（`mix`, `build-references`, `scan-references`, `learn-preferences`, `propagate-anchor`, `benchmark-ab`, `doctor`, `inspect-plugin`, `analyze-pair`, `inspect-song`）
- `src/mix_automation/pipeline.py`: v7 エンドツーエンド統合パイプライン（オフボ自動ステージング → 日英音素保護下処理 → オフボ較正＆好み込みターゲット合成 → GPU 微分 EQ＆CMA-ES → マルチセクション適応彫刻 → Stage ③ 5 クラス FX → Stage ④ ハモリ＆対旋律 MIDI → Studio One `.song` 出力）
- `src/mix_automation/analysis/relative_profile.py`: 曲非依存の対オケ相対音響プロファイル・アーキタイプ分類・アンカー伝播数理モデル
- `src/mix_automation/analysis/reference_extractor.py`: `BS-RoFormer` 分離、44 曲公式オフボ較正フィードバック、9-NN 合成、`youtube_references` 4 段階自動検品ゲート
- `src/mix_automation/analysis/preference_learner.py`: Studio One `.song` 手直し差分（フェーダー・センド・クリップゲイン・`Persistent Q`）抽出、EMA 好み学習、高度アンカー伝播
- `src/mix_automation/analysis/ab_harness.py`: `-14 LUFS` ラウドネス正規化 3 区間 A/B 比較クリップ抽出、Do-Not-Regress 自動監査、ブラインド X/Y 試聴 HTML コンソール生成
- `src/mix_automation/config/plugin_registry.py`: 実機検証済み 27 VST3 プラグインレジストリ
- `src/mix_automation/dsp/lyrics_aligner.py`: 日英混在歌詞 `.txt` パース（`pykakasi`）、母音コア／子音アタック境界分離、音響 DP＋`MMS_FA` アライメント
- `src/mix_automation/dsp/preprocess.py` & `pitch_timing.py`: `RX 10` 下処理、音素保護型ピッチ／タイミング微補正、生歌補正定量ベンチマーク、歌詞付きマーカー生成
- `src/mix_automation/dsp/vst3_adapter.py`: 厳格な VST3 型強制・読み戻し検証アダプタ（Unicode トークン正規化、`Persistent Q` 16 バンド、`Soothe2`、`LA-2A`、`VocalSynth 2`、`MAutoPitch`、`kHs Dual Delay`、`Trash 2` 等）
- `src/mix_automation/dsp/vst3_optimizer.py`: CUDA GPU 10 帯域微分彫刻 EQ、自動狭 Q ノッチ、マルチセクション 4 帯域ゼロ位相適応彫刻、自己相関検証付き連続テンポ追跡、Bark Mid アンマスキング、ハモリ／コーラス独立バス
- `src/mix_automation/creative/fx_and_harmony.py`: 本家 5 クラス FX 検出・適用、`Nectar 3` コード整合ハモリ、コール＆レスポンス／対旋律提案＆4 トラック MIDI 書き出し
- `src/mix_automation/io/audio_loader.py` & `song_builder.py` / `studio_one.py`: ステム自動分類、Studio One 5.5 ネイティブ `.song` / `.vstpreset` / `.envelopex` / 可変 `TempoMap` / スポット FX トラック生成
- `tests/test_core.py` & `tests/test_vst3_contracts.py`: 単体 18 件＋実機 VST3 契約 7 件（計 25 テスト）

## 5. ドキュメント＆スキル索引（必要時に読むこと）

- `docs/ROADMAP.md` — 4 段階ロードマップ原文・v7.0 到達度・残課題・次期 R&D
- `docs/ARCHITECTURE.md` — v7.0 詳細信号フロー・数理最適化・OSS アセット一覧
- `docs/CHANGELOG_AND_DECISIONS.md` — v1→v7.0 の全変遷、実曲フォレンジックと構造的解決、再発防止ルール
- `docs/VST3_AND_STUDIO_ONE_HANDBOOK.md` — 27 プラグイン実機仕様・22 種 Studio One ClassID・`.song` 構造
- `docs/BENCHMARKS_AND_REFERENCES.md` — 44 曲 DB 統計・`youtube_references` 検品仕様・全 4 曲 v7.0 ベンチマーク実測値
- `docs/reports/` — インタラクティブ HTML レポート（`stage4_roadmap.html`, `daw_3songs_console.html`, `ab_blind_test/index.html`）
- `.agents/skills/automix-workflow/SKILL.md` / `vst3-and-studio-one/SKILL.md` / `reference-db-pipeline/SKILL.md`
