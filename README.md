# MIX Automation v6 (`mix-automation`)

歌ってみた制作における**「どの楽曲・声域・編成でも破綻せずオケの内側へ溶け込む上級者クオリティ」**と**「Studio One 5.5 での真正 Steinberg `.vstpreset` 1対1再現・緻密な音作り支援」**を両立する自動ボーカルMIXエンジン（v6.0）です。

## プロジェクトドキュメント & Antigravity 構成

本リポジトリは Antigravity のプロジェクト **MIX_automation** から直接継続開発できるよう、設計・運用知見をすべて構造化して管理しています。

- [GEMINI.md](./GEMINI.md) — Antigravity プロジェクト憲法・設計不変条件・コードベース索引（プロジェクト内で自動読み込み）
- [docs/ROADMAP.md](./docs/ROADMAP.md) — ユーザーの 4 段階ロードマップ原文・現在地（`v6.0`）・未解決課題・次期 R&D 方針
- [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) — v6.0 信号フロー・数理最適化・OSS アセット一覧
- [docs/CHANGELOG_AND_DECISIONS.md](./docs/CHANGELOG_AND_DECISIONS.md) — v1→v6 の開発変遷・試聴フィードバック対応記録・再発防止アンチパターン
- [docs/VST3_AND_STUDIO_ONE_HANDBOOK.md](./docs/VST3_AND_STUDIO_ONE_HANDBOOK.md) — 実機検証済み 27 VST3 仕様・地雷・Studio One 5.5 ClassID と `.vstpreset` 復号仕様
- [docs/BENCHMARKS_AND_REFERENCES.md](./docs/BENCHMARKS_AND_REFERENCES.md) — 44 曲リファレンス DB 統計と 3 曲ベンチマーク実測結果
- [docs/reports/stage4_roadmap.html](./docs/reports/stage4_roadmap.html) / [docs/reports/daw_3songs_console.html](./docs/reports/daw_3songs_console.html) — インタラクティブ進捗＆DAW 風コンソール比較レポート
- `.agents/skills/` — Antigravity 運用スキル（`automix-workflow`, `vst3-and-studio-one`, `reference-db-pipeline`）

## v6.0 主要アーキテクチャ

1. **16バンド超高精細 `Persistent Q`（10帯域微分彫刻EQ＋自動狭Qノッチ＋3帯域Dynamic/Spectral EQ）**
   従来の6バンド静的EQを刷新し、歌唱F0連動のHPF（Band 1）、44曲固有声質プロファイルと対象曲のオケ補完マージン（`band_margins_db`）を同時最適化する10帯域ERB/オクターブ微分EQ（Bands 2–11、`z_db.detach()` による中低域スクープ防止機構付き）、声・部屋の定常共鳴を自動検出して除去する狭Qノッチ2本（Bands 12–13、`Q=6.5–7.2`）、さらに実時間3帯域 Dynamic / Spectral EQ（Band 14: 290Hz動的近接こもり制御、Band 15: 1.2ms超高速動的3.25kHzアタック抑制、Band 16: 7.8kHzスペクトル高域ジリジリ緩和）を単一の `Persistent Q` 内に統合し、Studio One 5 へ1対1ネイティブ書き出しします。
2. **高域ジリジリ物理根絶・アタック平滑化ダイナミクス＋Post-Comp 4帯域 `Soothe2`＋6帯域Bark子音針緩和**
   暗騒音を持ち上げていた `Waves MV2` 過剰アップワード圧縮と遅いアタック・過ドライブの1段目VCA設定を全面刷新。クリーンな高速アタック `Softube VCA Compressor`（`attack=0.52–0.72`、`output_drive=0.1`）で鋭い子音ピークを丸め、全アーキタイプでユニティゲイン自動較正済み `UAD Teletronix LA-2A` により暗騒音を持ち上げずに声の芯を滑らかに整えます。パラレルサチュレーション（`Softube Saturation Knob`）は `240 Hz–4.0 kHz` 中域限定バンドパスで高域歪みを物理遮断し、後段に4帯域較正済み `Oeksound Soothe2`（`3.1k` 刺激・`6.4k` 歯擦音・`10k+` ジリジリ高域シェル）と6帯域Bark子音トランジェントソフトシェイパーを配置しています。
3. **事前アンマスキング整合＆双方向 `Wet-Tail` 接着（声浮き防止）**
   14帯域Bark適応Midアンマスキング（`0.65–1.5 dB`）を最適化計算の前段に組み込み、音量・部分ラウドネスの二重持ち上がりを解消。同時に `Wet-Tail`（空間余韻比）を過多・過少の双方向でプロ目標へ揃え、ボーカルをオケの室内空間の内側へ定着させます。
4. **メインハモリ／コーラス分離処理＆ライセンス済み `iZotope VocalSynth 2` 統合**
   メインハモリ（`+3`/`-3`/`harm`）とコーラス（`choir`）を独立バスへ分離。メインハモリはリード声の中心フォルマント連動「芯抜きEQ」、`0.8 ms` 超高速アタック＋`340 ms` 低速リリース高レシオ圧縮による「語頭潰し」、`5 ms` 短プリディレイの奥行きリバーブ＋拡散マイクロディレイでリードの背後へ立体分離させつつ聴こえやすい音量（`-8.6 dB` 真ステレオK-LUFS）を確保します。また新規導入された `iZotope VocalSynth 2`（317パラメータ）をレジストリ・アダプタ・Studio One クラス定義へ統合し、Stage 3 のラジオボイス演出や合成ハモリ質感補正に活用します。

## 使い方（CLI）

```powershell
# 1. フル自動MIXの実行（44曲連続音響k-NN＋梓川テイスト25%ブレンド）
C:\MIX_automation\.venv\Scripts\mix-auto.exe mix "C:\MIX_automation\data\raw\ラブ&デストロイ" --style auto
C:\MIX_automation\.venv\Scripts\mix-auto.exe mix "C:\MIX_automation\data\raw\シックスフィートアンダー" --style auto
C:\MIX_automation\.venv\Scripts\mix-auto.exe mix "C:\MIX_automation\data\raw\Fall in Love" --style auto

# 純粋な44曲k-NN（テイストオフセットなし）や別MIX師テイスト指定:
C:\MIX_automation\.venv\Scripts\mix-auto.exe mix "C:\MIX_automation\data\raw\曲フォルダ" --taste none
C:\MIX_automation\.venv\Scripts\mix-auto.exe mix "C:\MIX_automation\data\raw\曲フォルダ" --taste sou --taste-weight 0.30

# 2. リファレンス全曲の再分離＆プロファイル再構築
C:\MIX_automation\.venv\Scripts\mix-auto.exe build-references

# 3. ローカル診断 & 全21テスト（単体＋実VST3契約テスト）実行
C:\MIX_automation\.venv\Scripts\mix-auto.exe doctor
C:\MIX_automation\.venv\Scripts\pytest.exe C:\MIX_automation\tests -v
```
