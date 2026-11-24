# MIX Automation — 最高品質アーキテクチャ設計書

## 1. エグゼクティブ・サマリーと前提条件

本プロジェクトは、歌ってみた制作における目標①〜④を段階的かつ最高品質で達成し、**「どの楽曲でも破綻しない上級者（商業・一流MIX師）レベルの混ざり具合」**と**「高頻度投稿を可能にする圧倒的な時短」**を実現するための専用システムである。

ローカル実機環境の調査結果および制作方針に基づき、以下の前提条件を設計の根幹に据える。

### ローカル実機リソース（確認済み）
- **GPU**: NVIDIA GeForce RTX 5060 Ti (16GB VRAM) — BS-RoFormer音源分離、SOFA音素アライメント、RMVPEピッチ解析、PyTorch微分可能DSP最適化をすべてローカル高速実行可能。
- **DAW**: PreSonus Studio One 5 (`5.5.2.86528`) — `.dawproject` は Studio One 6.5以降専用であるため採用せず、**Studio One 5 ネイティブの `.song`（ZIP + XML + Presets + Media）テンプレート注入方式**および **Steinberg標準 `.vstpreset` 出力方式**を採用する。
- **所有VST3プラグイン資産（確認済み主要モジュール）**:
  - **下処理（iZotope RX 10）**: `RX 10 Mouth De-click`, `RX 10 Breath Control`, `RX 10 De-plosive`, `RX 10 Voice De-noise`, `RX 10 Spectral De-noise`, `RX 10 De-ess`, `RX 10 De-reverb`
  - **EQ・ダイナミクス・質感**: `uaudio_teletronix_la-2a_tc` (UAD LA-2A), `uaudio_pultec_eqp-1a` / `meq-5` (UAD Pultec), `soothe2_x64` (Oeksound Soothe2), `ZL Spectrum Equalizer`, `Initial Dynamic EQ`, `TDR VOS SlickEQ`, `VCA Compressor`, `MCompressor`, `MAGC`, `Saturation Knob`, `iZotope Trash 2`
  - **空間・特殊効果・ピッチ・マスタリング**: `Rev PLATE-140`, `PhoenixVerb`, `R2`, `kHs Dual Delay`, `MDelay`, `MAutoPitch`, `iZotope VocalSynth 2`, `VocAlign6Standard`, `Melodyne`, `Ozone 12 Elements`, `WaveBreaker`, `SPAN`

---

## 2. コア設計思想（なぜ他曲でも混ざり具合が破綻しないのか）

1. **「上級者MIXの対オケ相対音響プロファイル」を目標関数化**
   ご自身の過去MIXを天井にするのではなく、理想とする一流MIX師・商業歌ってみた音源（複数曲）をターゲットとする。その際、従来のMP3逆相引き算（位相ノイズやマスタリングリミッター歪みが混入する悪手）を全廃し、`BS-RoFormer` / `Mel-Band RoFormer` による高精度音源分離と2mix全体の解析を組み合わせ、曲調に依存しない不変量である**「オケの帯域エネルギーに対するボーカルの突出マージン」「対オケ・クレストファクター比（張り付き度）」「2〜5kHzの動的共鳴ピーク上限（刺さらない滑らかさ）」**を数理モデル化する。
2. **決定論的DSP・VST3最適化エンジンとLLMディレクターの完全分離**
   0.1dB・ミリ秒単位の精度を要する下処理・均し・EQ・コンプ・レゾナンス抑制（目標①・②）にはLLMを一切介在させず、数理最適化エンジンが100%決定論的に計算する。Gemini / Claude 等のLLMは、歌詞解釈を伴う空間・特殊演出の配置判断（目標③）、ハモリ・対旋律のアレンジ提案（目標④）、および人間の感覚的な修正要望（「もう少し声を前に張り付かせたい」等）を数理エンジンの目標重みへ変換するインターフェースに専念させる。
3. **1パートの微調整が全パートへ波及する「アンカー伝播（Anchor Propagation）」**
   Aメロとサビなどでオケの音圧・密度が大きく変化する楽曲において、Studio One上でユーザーが1つの基準パート（例：サビ）の設定だけを納得いくまで微調整すれば、システムがそのパートの「修正後の対オケ相対バランス」をアンカーとして計測し、他パートのオケとのスペクトル密度差・短期LUFS差を打ち消すように全セクションのゲイン・Threshold・動的EQオフセットを自動再計算する。

---

## 3. 統合・昇華する既存GitHub優良アセット

車輪の再発明を避け、以下の実績あるオープンソース基盤を統合して昇華させる。

- **[spotify/pedalboard](https://github.com/spotify/pedalboard)**: 実機VST3プラグイン（RX 10, UAD, Soothe2等）のロード、高速オーディオレンダリング、パラメータ操作、および `raw_state` バイナリ抽出。
- **[nomadkaraoke/python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)**: UVR / `BS-RoFormer` / `Mel-Band RoFormer` によるリファレンス音源からのクリーンなボーカル・ハモリ・オフボーカル分離。
- **[csteinmetz1/dasp-pytorch](https://github.com/csteinmetz1/dasp-pytorch)** & **[SonyResearch/diffvox](https://github.com/SonyResearch/diffvox)**: PyTorch上の微分可能EQ・コンプレッサー・サチュレーターによる高速な第1段階パラメータ勾配探索。
- **[qiuqiao/SOFA](https://github.com/qiuqiao/SOFA)**: 歌声特化型フォーストアライナー（日本語・英語対応）による音素（母音・子音・歯擦音・ブレス）ミリ秒境界特定。
- **[haidog-yaqub/DiffPitcher](https://github.com/haidog-yaqub/DiffPitcher)** / **RMVPE + WORLD/PSOLA**: しゃくりやビブラートを保護しながら母音中心部を安定させる安全圏ピッチ補正。
- **[mhrice/RemFx](https://github.com/mhrice/RemFx)** & **[nuniz/blind_rt60](https://github.com/nuniz/blind_rt60)**: 本家ボーカルの特殊エフェクト・残響変化のタイムライン検出。
- **[mitmedialab/ai-harmonizer-nime2025](https://github.com/mitmedialab/ai-harmonizer-nime2025)**: コード進行・主旋律解析とボイスリーディング規則に基づくハモリ・コーラス譜面生成および声質変換ガイド生成。

---

## 4. 段階的開発ロードマップ

### Phase 1（目標①＋Studio One 5連携基盤）: 混ざり具合の完全汎用化と `.song` / `.vstpreset` 出力
1. 上級者リファレンス音源群からの「対オケ相対音響プロファイル（Target Mix Profile）」抽出器の構築。
2. 二段階ハイブリッド最適化（微分可能DSPによる理想カーブ算出 → `pedalboard` + CMA-ESによる実機VST3パラメータ最適化）の実装。
3. Studio One 5 テンプレート `.song` 注入ビルダーおよび `.vstpreset` 生成器、ならびにパート間「アンカー伝播」機構の実装。

### Phase 2（目標②）: RX 10統合下処理・音素連動ゲインライディング・全自動チェイン完成
1. `RX 10 Mouth De-click` / `De-plosive` / `Voice De-noise` / `Breath Control` の自動チェイン化と、声質劣化ゼロのオフライン下処理。
2. `SOFA` による日英音素境界解析と対オケ短期LUFSを組み合わせた「音素単位プリ・ゲインライディング（太い土台作り）」＋安全圏ピッチ補正の実装。
3. `UAD LA-2A` + `Pultec` + `Soothe2` + `Initial Dynamic EQ`（対オケ動的アンマスキング）＋マスタリング（`Ozone 12 Elements` + `WaveBreaker`）までの一貫ワンボタン処理化。

### Phase 3（目標③）: 歌詞・秒単位の本家FXスタイル検出・自動再現
1. 本家音源の分離ボーカルと歌詞タイムラインから、ラジオボイス（帯域制限＋歪み）、ケロケロ（`MAutoPitch` 高速Retune）、リバーブ投げ・ディレイ飛ばし（`Rev PLATE-140` / `PhoenixVerb` / `kHs Dual Delay` センド急増）をイベント検出。
2. Studio One 5 の `.song` オートメーションレーンおよびFXトラックへ秒・歌詞単位で自動配置。

### Phase 4（目標④）: AIディレクターによるコーラス・ハモリ提案＆本人声ガイド生成
1. 本家ハモリ分離またはコード進行解析からの複数ハモリ・対旋律MIDI生成。
2. ユーザー自身の歌声モデル（軽量RVC等）を通した仮歌ガイドWAVの自動生成と、`.song` 内のガイドトラックへの自動配置。
