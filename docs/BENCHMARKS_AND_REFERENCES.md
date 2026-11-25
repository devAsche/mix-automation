# リファレンスデータベース & v7.0 全 4 曲ベンチマーク検証記録

## 1. 44 曲プロリファレンスデータベース & `youtube_references` 自動検品基盤

- **音源分離モデル**: `model_bs_roformer_ep_317_sdr_12.9755.ckpt`（GPU ローカル実行・`data/processed/roformer_stems/` キャッシュ）
- **収録規模**: 8 歌い手（`AokiYo` 5, `aoru` 8, `Azusagawa` 5, `HoshirubeSho` 5, `NanadakiIma` 9, `Saburou` 5, `Sou` 2, `TOY` 5）、合計 **44 曲**（全 44 曲に公式オフボーカルをペアリング済み）
- **アーキタイプ構成**: `dense_rock_edm` (28), `groovy_mid_pop` (11), `intimate_ballad_sparse` (5)
- **公式オフボ較正フィードバック**: `median_sep_minus_offvocal_lufs_db = -1.16 dB`, `tilt = +0.28 dB`, `crest = -0.21 dB` を `synthesize_acoustic_knn_target` の微小バイアス補正へ統合済み（高域ブースト上限 `<= 0.0 dB` 安全ガード付き）。
- **`youtube_references` 受け入れ・自動検品ゲート（`mix-auto scan-references`）**:
  - `C:\MIX_automation\youtube_references\_info\existing_44_songs.json` による既存 44 曲重複ブロック。
  - 15.5 kHz 未満ブリックウォール劣化・クリッピング・モノラル崩壊検査、および 12 半音 CQT クロマ回転相関による原曲キー一致 vs キー変更（`key_shift_semitones`）自動判定を実装済み。

---

## 2. v7.0 全 4 曲ベンチマーク＆Do-Not-Regress 実測結果（`2026-11-26` 検証済み）

`ruff` 0 エラー・`pytest` **25/25 PASS**（単体 18＋実機 VST3 契約 7）の状態で全 4 曲を独立 CLI プロセスでレンダリングし、`mix-auto benchmark-ab` による `-14.0 LUFS` 正規化ブラインド A/B コンソール（[docs/reports/ab_blind_test/index.html](./reports/ab_blind_test/index.html)）および Studio One 5.5 監査で **4 曲すべて ALL PASS（JUCE リーク `0`）** を達成。

### 2.1 ラブ&デストロイ (`data/outputs/ラブandデストロイ_auto/`) — **ALL PASS**
- **BPM / Key**: `139.1 BPM` / `E Minor`
- **K-LUFS V/I**: `+2.18 dB`（目標 `+2.05 dB`、誤差 `0.12 dB` PASS）
- **Wet-Tail**: `-24.08 dB`（目標 `-24.08 dB`、誤差 `0.00 dB`）
- **固有声質 (`Body / Pres / Sib / Air`)**: `-4.42 / -11.82 / -19.36 / -21.85 dB`（中低域スクープなし・高域ジリつきゼロ）
- **歌詞・演出・ディレクション**: 歌詞 `50` 行同期、Stage ③ `distortion_grit` 検出・Studio One スポットトラック注入、対旋律ガイド `23` ノート（4 トラック `.mid` 出力）
- **Studio One 5.5 監査**: XML 正常 / ネイティブ `.vstpreset` `17` 個 / 細分化イベント `5` 個 / JUCE リーク `0`

### 2.2 シックスフィートアンダー (`data/outputs/シックスフィートアンダー_auto/`) — **ALL PASS**
- **BPM / Key**: `127.5 BPM`（ポリリズム倍音ジャンプ抑制により原曲テンポへ完全安定化） / `G Minor`
- **K-LUFS V/I**: `+1.06 dB`（目標 `+1.12 dB`、誤差 `0.07 dB` PASS） / **Mid K-LUFS V/I**: `+2.11 dB`（目標 `+2.01 dB`）
- **Side/Mid 幅**: `-9.34 dB`（目標 `-9.11 dB`） / **Wet-Tail**: `-23.80 dB`（目標 `-23.06 dB`）
- **固有声質 (`Body / Pres / Sib / Air`)**: `-6.26 / -10.40 / -21.35 / -23.20 dB`
- **歌詞・演出・ディレクション**: 歌詞 `67` 行同期、Stage ③ 本家検証 FX `9` 件（`radio_voice` [`VocalSynth 2`]、`hard_autotune` [`MAutoPitch`]、`distortion_grit` [`Trash 2`]、`spot_delay_throw` [`kHs Dual Delay`] を歌詞スニペット付きで排他 Studio One トラック注入）、`choir_wide_bus` 独立出力、対旋律ガイド `44` ノート（4 トラック `.mid` 出力）
- **Studio One 5.5 監査**: XML 正常 / ネイティブ `.vstpreset` `19` 個 / 細分化距離イベント `20` 個 / JUCE リーク `0`

### 2.3 Fall in Love (`data/outputs/Fall_in_Love_auto/`) — **ALL PASS**
- **BPM / Key**: `122.6 BPM` / `G Minor`
- **Mid K-LUFS V/I**: `+2.42 dB`（目標 `+2.46 dB`） / **K-LUFS V/I**: `+1.21 dB`（目標 `+2.02 dB`、誤差 `0.81 dB` PASS）
- **固有声質 (`Body / Pres / Sib / Air`)**: `-4.99 / -11.64 / -24.55 / -26.35 dB`（v6 境界付近だった Body も `-4.99 dB` で余裕の PASS、高域ジリつき完全ゼロ）
- **ハモリ・ディレクション**: モノラルハモリ 1 本の対称 `11 ms` Haas 展開＋芯抜き／語頭潰し、対旋律ガイド `47` ノート（4 トラック `.mid` 出力）
- **Studio One 5.5 監査**: XML 正常 / ネイティブ `.vstpreset` `16` 個 / 細分化距離イベント `14` 個 / JUCE リーク `0`

### 2.4 エバ (`data/outputs/エバ_auto/`) — **新規検証・ALL PASS**
- **素材特性と自動ステージング**: 未マスタリング極小音量オフボ（`-36.3 LUFS` / `-38.4 dB` RMS）を検出し、トゥルーピーク保護付きで `+19.65 dB` 自動ヘッドルームステージングを実行。実録ハモリ 0 本のフラット配置構成。
- **BPM / Key**: `140.0 BPM`（原曲テンポ一致） / `D Minor`
- **Mid K-LUFS V/I**: `+2.26 dB`（目標 `+2.04 dB`） / **K-LUFS V/I**: `+0.61 dB`（目標 `+1.18 dB`、誤差 `0.57 dB` PASS）
- **Wet-Tail**: `-22.00 dB`（目標 `-21.92 dB`、誤差 `0.08 dB`）
- **固有声質 (`Body / Pres / Sib / Air`)**: `-5.06 / -10.25 / -21.41 / -23.70 dB`
- **Stage ④ 合成ハモリ自動採用＆MIDI**: 実録ハモリ不在のため `iZotope Nectar 3`（440 ノート、コード衝突回避 116 箇所）＋ `VocalSynth 2` Biovox グロスを 2mix に自動採用し、Studio One 上でもアンミュート配置＋対旋律ガイド `12` ノート（4 トラック `.mid`）出力。
- **Studio One 5.5 監査**: XML 正常 / ネイティブ `.vstpreset` `16` 個 / 細分化距離イベント `21` 個 / JUCE リーク `0`
