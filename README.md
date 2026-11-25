# MIX Automation v5 (`mix-automation`)

歌ってみた制作における**「どの楽曲・声域・編成でも破綻しない上級者クオリティの混ざり具合」**と**「Studio One 5.5 での真正 Steinberg `.vstpreset` 1対1再現・時短ディレクション支援」**を両立する自動ボーカルMIXエンジン（v5）です。

## v5 主要アーキテクチャ

1. **44曲プロ完成音源×公式オフボーカル較正「連続音響 k-NN」＋梓川テイストアンカー**
   8組・計44曲（ロック／ボカロEDM／グルーヴポップス／バラード／スパースポップ）を `BS-RoFormer` で分離し、全44曲の公式オフボーカル音源とペア較正（`schema_version: 4.0`）。フル尺オケ密度・ダイナミックレンジ・ステレオ幅・高域ロールオフ・歌唱F0から音響的に最も近い9曲を連続合成し、標準で好みのMIX師テイスト（`--taste azusagawa --taste-weight 0.25`、解除は `--taste none`）を滑らかに統合します。
2. **グルーヴ保護ピッチ／タイミング制御＆自己較正・アタック保護ディエッシング**
   Melodyne 補正済み・人間グルーヴのテイクに対しては無断のタイミングスリップやボコーダ再処理を一切行わず、確認箇所を Studio One マーカーとして配置します（未補正テイク向け `--auto-pitch` は Praat `PitchTier` 連続オーバーラップ加算でビブラート・子音を保護、タイミング自動調整は `--auto-align` で明示オプトイン）。下処理は `-18.0 LUFS` ゲインステージング後に `RX 10 Voice De-noise`（`'Music'`／`'Gentle'`）＋広域二分探索の `RX 10 Spectral De-ess`（`5.8 kHz`、子音アタック `1.8〜4.2 kHz` 保護）を実行します。
3. **連続距離トラッカー＆14帯域Bark適応スペクトル空間確保＆真ステレオK-LUFSハモリバス**
   上下両方向のセクション変化検出に加えて、オケのボーカル競合帯域（`350 Hz–7.5 kHz`）を連続追従する距離トラッカーを導入。サブベースを残したままギターやハイハットが抜ける静かなブレイク／落ちサビでも、音量・パラレル倍音・高域エッジ緩和・ルーム接着を滑らかに自動連動させます。録音済みハモリ・コーラスは真のステレオK特性ラウドネス基準（`-11.2 dB`）＋歯擦音緩和＋共有ルーム／プレート接着で自然に馴染ませ、録音ハモリがある曲では `Nectar 3` を自動ミュートしてガイド専用に徹します。
4. **Studio One 5.5 真正 Steinberg `.vstpreset` 1対1動的注入（JUCE→Steinberg 変換）**
   Pedalboard の JUCE `<VST3PluginState>` ラッパーからネイティブ Steinberg `Comp`／`Cont` チャンクを完全復号し、チャンネル名一致（`メイン`／`Saturation Parallel`／`Phoenix EarlyRoom`）・クリーンなハモリインサート構成（旧テンプレート `C1`/`L1`/`Doubler2` 除去）・展開済みミキサーコンソールで Studio One 5 上でもそのまま全ノブ設定が実機再現されます。`stems/` サブフォルダ形式とフラットフォルダ形式の両方を自動認識します。

## 使い方（CLI）

```powershell
# 1. フル自動MIXの実行（44曲連続音響k-NN＋梓川テイスト25%ブレンド）
C:\MIX_automation\.venv\Scripts\mix-auto.exe mix "C:\MIX_automation\data\raw\ラブ&デストロイ" --style auto
C:\MIX_automation\.venv\Scripts\mix-auto.exe mix "C:\MIX_automation\data\raw\シックスフィートアンダー" --style auto
C:\MIX_automation\.venv\Scripts\mix-auto.exe mix "C:\MIX_automation\data\raw\Fall in Love" --style auto

# 純粋な44曲k-NN（テイストオフセットなし）や別MIX師テイスト指定:
C:\MIX_automation\.venv\Scripts\mix-auto.exe mix "C:\MIX_automation\data\raw\曲フォルダ" --taste none
C:\MIX_automation\.venv\Scripts\mix-auto.exe mix "C:\MIX_automation\data\raw\曲フォルダ" --taste sou --taste-weight 0.30

# 2. リファレンス全曲（manifest*.json自動検出）の再分離＆プロファイル再構築
C:\MIX_automation\.venv\Scripts\mix-auto.exe build-references

# 3. ローカル診断 & 全19テスト実行
C:\MIX_automation\.venv\Scripts\mix-auto.exe doctor
C:\MIX_automation\.venv\Scripts\pytest.exe C:\MIX_automation\tests -v
```
