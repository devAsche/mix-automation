# MIX Automation v2 (`mix-automation`)

歌ってみた制作における**「どの楽曲・声域でも破綻しない上級者クオリティの混ざり具合」**と**「高頻度投稿のための時短・ディレクション支援」**を両立する、PreSonus Studio One 5.5 ネイティブ対応の自動ボーカルMIXエンジン（v2）です。

## v2 主要アーキテクチャ（目標①〜④ 一貫対応）

1. **プロ35曲×BS-RoFormer GPU分離による「声域×オケ編成 層別プロファイル」最適化（目標①）**
   7組・計35曲のプロ完成音源を `BS-RoFormer`（SDR 12.97）でGPU音源分離し、ボーカルの声域（`low_mid` / `high`）とオケ編成アーキタイプ（`dense_rock_edm` / `groovy_mid_pop` / `intimate_ballad_sparse`）ごとに「対オケ6帯域マージン」「クレスト比」「短期位置安定度」「歌詞明瞭度指数（2〜8Hz変調深度）」「空間余韻比」をターゲット化。実機の商用VST3チェーンを直接ループ内で駆動しながらCMA-ESで最適解を探索します。
2. **楽曲の編成に応じた商用VST3マルチチェーン自動選択＆ディレイ主軸の空間設計**
   固定1チェーンではなく、ロック・高密度ボカロには `Softube VCA` + `Waves MV2` + `Saturation Knob`、グルーヴィーなポップスには `Softube VCA` + `UAD LA-2A` + `UAD Pultec EQP-1A`、親密なバラードには `UAD LA-2A` + `TDR SlickEQ (Silky)` を `Persistent Q`（Dynamic EQ併用）と組み合わせて自動構成します。空間系はボワン感を排し、帯域制限スラップバックディレイ（78/92ms）＋ボーカルダッキング付き1/8ディレイ＋無変調ショートPlate（`R2` 0.92〜1.25s）でオケの音場に自然に接着します。
3. **ケロケロ感ゼロのクリーン下処理・スリップ同期・クローズドループ検品（目標②）**
   位相ボコーダによるピッチシフトを完全排除。Melodyne補正済みステムは自動判定して再ボコーダ処理をスキップし、未補正テイクのみ `Praat TD-PSOLA` でフォルマントとビブラートを保ったまま持続音を微補正します。さらに本家ボーカルとの無伸縮フレーズ・スリップ同期、実機 `iZotope RX 10` 下処理＋`Oeksound Soothe2`、`-18.0 LUFS` 標準ゲインステージング、ハモリ合算後のクローズドループ・バス検品、2mixバスグルー（リミッター駆動量上限3.2dB）まで一貫処理します。
4. **本家ボーカル根拠に基づく局所特殊演出（目標③）＆ Nectar 3 コード追従ハモリ生成（目標④）**
   本家原曲の分離ボーカルを照合し、実際に本家でロングリバーブ投げやラジオボイス演出が存在する箇所のみピンポイントで再現します。ハモリは実機 `iZotope Nectar 3` を駆動し、借用和音・転調による不協和（短2度・三全音衝突）をオケの局所CQTクロマグラムからノート単位で自動回避した上下ハモリを生成。完パケMIXとしてそのまま馴染むと同時に、Studio One上でソロにすれば録音用ガイドとしても機能します。

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

# 3. リファレンス35曲のGPU分離＆層別プロファイル再構築
C:\MIX_automation\.venv\Scripts\mix-auto.exe build-references

# 4. ローカルのStudio One 5・VST3プラグイン（全26種）稼働診断
C:\MIX_automation\.venv\Scripts\mix-auto.exe doctor
```
