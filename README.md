# MIX Automation v4 (`mix-automation`)

歌ってみた制作における**「どの楽曲・声域・編成でも破綻しない上級者クオリティの混ざり具合」**と**「Studio One 5.5 での1対1実機再現・時短ディレクション支援」**を両立する自動ボーカルMIXエンジン（v4）です。

## v4 主要アーキテクチャ

1. **44曲プロ完成音源×公式オフボーカル較正による「連続音響 k-NN」ターゲット合成**
   8組・計44曲（ロック／ボカロEDM／グルーヴポップス／バラード／スパースポップ）を `BS-RoFormer` で分離し、全44曲の公式オフボーカル音源とペア較正（`schema_version: 4.0`）。フル尺のオケ密度・ダイナミックレンジ・クレスト・ステレオ幅・高域ロールオフ・歌唱F0から音響的に最も近いプロ楽曲を距離重み付き k-NN で連続合成し、平歌・落ちサビでの近接シフト量までデータから学習します。`--taste <artist> --taste-weight 0.35` による好みのMIX師テイスト合成にも対応します。
2. **実測検証付きVST3契約アダプタ＆自己較正・アタック保護ディエッシング**
   全商用VST3のパラメータ書き込みを厳格な読み戻し検証（`dsp/vst3_adapter.py`）で保証し、実オーディオを用いた5系統のライブVST3契約テストで常時検証します。下処理は `-18.0 LUFS` 標準ゲインステージング後に `RX 10 Voice De-noise`（`'Music'`／`'Gentle'`）と広域二分探索の `RX 10 Spectral De-ess`（`5.8 kHz`、子音アタック帯域 `1.8〜4.2 kHz` 保護ガード付）を実行します。
3. **3段階ハイブリッド最適化・14帯域Bark適応スペクトル空間確保・セクション連動オートメーション**
   PyTorch 微分可能6バンドEQウォームスタート → 実機 `Softube VCA`＋第2段レベラー（`Waves MV2`／`UADx LA-2A`）の実測GR二分探索 → 5次元実機VST3 CMA-ES による心理音響仕上げを実施。オケ側には Side 成分を一切傷つけない14帯域 Bark 適応 Mid スペクトル空間確保（Trackspacer/Soothe2型）を適用し、さらにセクション連動のパラレル倍音・空間オートメーションとハモリ優先クローズドループ検品でリードの芯と抜けを両立します。
4. **Studio One 5.5 実機VST3 1対1動的注入（`io/song_builder.py`）**
   Studio One 5 側の正式な VST3 `classID` と Steinberg `.vstpreset` バイナリ合成により、`mainvocal`（`Persistent Q` → `VCA Compressor` → `MV2 Mono` または `UADx LA-2A` → `UADx Pultec EQP-1A` / `TDR VOS SlickEQ`）、専用FXチャンネル（`Saturation Parallel`・`Phoenix EarlyRoom`・`R2`・`H-Delay`・`Doubler2`）、およびマスターバス（`VCA Compressor`＋`Wave Breaker`）まで1対1で `.song` へ直接注入します。

## 使い方（CLI）

```powershell
# 1. フル自動MIXの実行（44曲連続音響k-NNで最適ターゲット＆実機VST3チェーンを自動構成）
C:\MIX_automation\.venv\Scripts\mix-auto.exe mix "C:\MIX_automation\data\raw\ラブ&デストロイ" --style auto
C:\MIX_automation\.venv\Scripts\mix-auto.exe mix "C:\MIX_automation\data\raw\シックスフィートアンダー" --style auto

# 好みのMIX師テイストを音響k-NNへ35%ブレンドする場合:
# (--style / --taste: azusagawa | toy | saburou | aokiyo | aoru | hoshirubesho | nanadakiima | sou)
C:\MIX_automation\.venv\Scripts\mix-auto.exe mix "C:\MIX_automation\data\raw\曲フォルダ" --style auto --taste azusagawa --taste-weight 0.35

# 2. Studio One 5 でアンカーパートを手直しした後、全パートへ再伝播
C:\MIX_automation\.venv\Scripts\mix-auto.exe propagate-anchor "C:\MIX_automation\data\outputs\...\AutoMIX.song" --anchor 2

# 3. リファレンス全曲（manifest*.json自動検出）の再分離＆v4.0プロファイル再構築
C:\MIX_automation\.venv\Scripts\mix-auto.exe build-references

# 4. ローカル診断 & 全16テスト実行
C:\MIX_automation\.venv\Scripts\mix-auto.exe doctor
C:\MIX_automation\.venv\Scripts\pytest.exe C:\MIX_automation\tests -v
```
