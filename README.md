# MIX Automation v3 (`mix-automation`)

歌ってみた制作における**「どの楽曲・声域でも破綻しない上級者クオリティの混ざり具合」**と**「高頻度投稿のための時短・ディレクション支援」**を両立する、PreSonus Studio One 5.5 ネイティブ対応の自動ボーカルMIXエンジン（v3）です。

## v3 主要アーキテクチャ（構造的欠陥の根絶＋聴覚心理最適化）

1. **実測検証付きVST3契約アダプタ（`dsp/vst3_adapter.py` & `tests/test_vst3_contracts.py`）**
   プラグインごとに異なる型・文字列列挙・負のスレッショルド仕様（`Softube VCA Compressor` の負値スレッショルドと `'4.0:1'`、`Softube Saturation Knob` の文字列 `'2.5'` / `'KEEP LOW'`、`UAD LA-2A` の `'35'`、`PhoenixVerb` の `' 8.0 Meters'` 等）を厳格な読み戻し検証（`set_vst3_param_strict`）で統一制御し、無音でのパラメータ設定失敗をゼロ化しています。実オーディオを用いた5系統のライブVST3伝達テストで常時検証します。
2. **自己較正・子音アタック保護ディエッシング（`dsp/preprocess.py`）**
   `-18.0 LUFS` ゲインステージング後に `RX 10 Spectral De-ess`（カットオフ `5.8 kHz`、スペクトラル動作）を5段階二分探索で自動較正します。歯擦音バースト帯域（`5.8–10.5 kHz`）のみ約 `2.4 dB` のリダクションに収め、子音アタック帯域（`1.8–4.2 kHz`）への影響を `0.5 dB` 未満に保護します。`RX 10 Voice De-noise` もボーカル向けの `'Music'` / `'Gentle'` モードで動作します。
3. **楽曲非依存の聴覚心理最適化＆公式オフボーカル較正（`analysis/relative_profile.py`, `reference_extractor.py`）**
   35曲のプロ完成カバー＋全35曲の公式オフボーカル音源のペア較正（`schema_version: 3.0`）により、サブベース量に左右されない `ITU-R BS.1770 K-Weighted LUFS`・`Mid K-LUFS`・18帯域 Bark/ERB 部分ラウドネス（マスキング余裕度）・7帯域ボーカル固有スペクトル形状（`vocal_intrinsic_bands_db`）をターゲット化。オケ固有の偏りに引っ張られてボーカル中域が破綻する構造的欠陥を解消しました。
4. **3段階ハイブリッド最適化（微分可能EQウォームスタート → 実測GR二分探索 → 5次元実機VST3 CMA-ES）**
   PyTorch による微分可能6バンド RBJ バイカッドEQウォームスタートで固有スペクトル形状＋オケ高域ロールオフ親和性（`0.22x`）を高速収束させた後、実機 `Softube VCA`（目標GR約 `3.5–4.0 dB`）と第2段レベラー（`Waves MV2` / `UAD LA-2A`）を実測二分探索でキャリブレーション。さらにオケ密度連動の並列 `Softube Saturation Knob`、3層空間設計（`PhoenixVerb` 8m初期反射ルーム＋スラップバック＋ダッキング1/8ディレイ＋無変調ショートPlate `R2`）を5次元 CMA-ES で仕上げます。
5. **音響重複排除・汎用ハモリ分類・専用ワイドハモリバス（`io/audio_loader.py` & `dsp/vst3_optimizer.py`）**
   同一フォルダに新旧モノラル／ステレオ書き出しが混在していても波形相関（`r >= 0.985`）で旧ステレオ重複を自動除外し、`+3` / `-3` / `上ハモ` / `下ハモ` / `choir` / `suboc` / `superoc` 等の日英ファイル名規則＋ピッチ差分フォールバックで正確に役割分類。パン振り前に全ステムを個別アクティブRMS正規化し、専用ワイドハモリバス（L/Rパン＋微細Haas広がり＋`Nectar 3` コード衝突回避生成）でプロ水準のステレオ音場を構築します。

## 使い方（CLI）

```powershell
# 1. フル自動MIXの実行（声域×オケ編成を自動判定して最適プロファイル＆VST3チェーンを選択）
# 明示スタイル指定も可: --style auto | consensus | azusagawa | toy | saburou | aokiyo | aoru | hoshirubesho | nanadakiima
C:\MIX_automation\.venv\Scripts\mix-auto.exe mix "C:\MIX_automation\data\raw\ラブ&デストロイ" --style auto
C:\MIX_automation\.venv\Scripts\mix-auto.exe mix "C:\MIX_automation\data\raw\シックスフィートアンダー" --style auto

# 未補正素音源に対してPraat TD-PSOLAピッチ微補正を強制適用する場合:
C:\MIX_automation\.venv\Scripts\mix-auto.exe mix "C:\MIX_automation\data\raw\曲フォルダ" --auto-pitch

# 2. Studio One 5 で1つのパート（アンカー）のクリップゲインを手直しした後、全パートへ再伝播
C:\MIX_automation\.venv\Scripts\mix-auto.exe propagate-anchor "C:\MIX_automation\data\outputs\...\AutoMIX.song" --anchor 2

# 3. リファレンス35曲＋公式オフボーカル較正による v3.0 プロファイル再構築
C:\MIX_automation\.venv\Scripts\mix-auto.exe build-references

# 4. ローカルのStudio One 5・VST3プラグイン稼働診断 & テスト実行
C:\MIX_automation\.venv\Scripts\mix-auto.exe doctor
C:\MIX_automation\.venv\Scripts\pytest.exe C:\MIX_automation\tests -v
```
