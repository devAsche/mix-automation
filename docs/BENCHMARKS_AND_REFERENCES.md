# リファレンスデータベース & v6.0 ベンチマーク検証記録

## 1. 44 曲プロリファレンスデータベース（`profiles/pro_target_profiles.json`, `schema_version: 4.0`）

- **音源分離モデル**: `model_bs_roformer_ep_317_sdr_12.9755.ckpt`（GPU ローカル実行・`data/processed/roformer_stems/` キャッシュ）
- **収録規模**: 8 歌い手（`AokiYo`, `aoru`, `Azusagawa`, `HoshirubeSho`, `NanadakiIma`, `Saburou`, `Sou`, `TOY`）、合計 **44 曲**（全 44 曲に公式オフボーカルをペアリング済み）
- **アーキタイプ構成**:
  - `dense_rock_edm`: **28 曲**
  - `groovy_mid_pop`: **11 曲**
  - `intimate_ballad_sparse`: **5 曲**（※今後の `muse` 大規模収集で優先拡充したい疎領域）
- **公式オフボ較正統計（`official_offvocal_calibration`）**:
  - `num_paired_official_offvocals`: `44`
  - `median_master_minus_offvocal_lufs_db`: `+1.94 dB`
  - `median_sep_minus_offvocal_lufs_db`: `-1.16 dB`
  - `median_sep_minus_offvocal_crest_db`: `-0.21 dB`
  - `median_sep_minus_offvocal_tilt_db`: `+0.28 dB`
- **ターゲット解決の既定動作**:
  - `--style auto`: 対象曲のオケ音響記述子＋歌唱 F0 中央値から 44 曲の連続 9-NN 補間を実施。
  - `--taste azusagawa --taste-weight 0.25`: ユーザー好みのテイスト基準（梓川）を 25% ブレンド（`--taste none` で純粋 44 曲 9-NN）。
- **収録スタイルキー（20 種）**: `consensus`, `register_low_mid`, `register_high`, 各声域×3 アーキタイプ、`azusagawa`, `toy`, `saburou`, `aokiyo`, `aoru`, `hoshirubesho`, `nanadakiima`, `sou`。

---

## 2. v6.0 3 曲ベンチマーク実測結果（`2026-11-25` 検証済み）

全 3 曲とも `ruff` 0 エラー・`pytest` 21/21 PASS の状態でレンダリングおよび Studio One 5.5 XML / `.vstpreset` 監査を完了。インタラクティブな DAW 風コンソール比較は [docs/reports/daw_3songs_console.html](./reports/daw_3songs_console.html) を参照。

### 2.1 ラブ&デストロイ (`data/outputs/ラブandデストロイ_auto/`)
- **K-LUFS V/I**: `+2.00 dB`（目標 `+2.00 dB`）
- **Wet-Tail**: `-24.13 dB`（目標 `-24.08 dB`）
- **2–6 kHz アタック指標 (`atk26`)**: `13.54`（プロ基準 `13.50`）
- **歯擦音 (`sib`)**: `-19.83 dB`（梓川基準 `-19.01 dB`）
- **自動狭 Q ノッチ**: `[880 Hz, 388 Hz]`
- **Studio One 5.5 監査**: XML 正常 / ネイティブ `.vstpreset` 17 個 / JUCE リーク `0`

### 2.2 シックスフィートアンダー (`data/outputs/シックスフィートアンダー_auto/`)
- **K-LUFS V/I**: `+1.23 dB`（目標 `+1.07 dB`）
- **Side/Mid 幅**: `-9.14 dB`（目標 `-9.11 dB`）
- **Presence マージン**: `+1.02 dB`（目標 `+0.72 dB`、v5 の `+4.19 dB` から大幅鎮静化）
- **Sibilance マージン**: `-3.54 dB`（目標 `-3.56 dB`）
- **2–6 kHz アタック指標 (`atk26`)**: `13.51`（プロ基準 `13.50`）
- **静寂部高域フロア (`qHF`)**: `-57.88 dB`（v5 の `MV2` 起因ノイズ上昇から `4.8 dB` 改善）
- **セクション・特殊演出・バス**: 14 セクション（2:15 静寂ドロップ `133.8–157.2s` 追従済み）、`VocalSynth 2` ラジオボイスを `137.44–138.58s` へ自動適用（重複 11 件は自動スキップ）、`choir_wide_bus` 独立出力
- **Studio One 5.5 監査**: XML 正常 / ネイティブ `.vstpreset` 16 個 / JUCE リーク `0`

### 2.3 Fall in Love (`data/outputs/Fall_in_Love_auto/`)
- **Mid K-LUFS V/I**: `+2.63 dB`（目標 `+2.41 dB`）
- **固有声質 Presence**: `-11.21 dB`（目標 `-11.26 dB`）
- **Air マージン**: `-2.42 dB`（v5 の過剰 `+8.16 dB` から解消）
- **高域／中域比 (`hf_mid`)**: `-20.28 dB`（v5 の `-13.48 dB` からジリつき解消）
- **静寂部高域フロア (`qHF`)**: `-61.01 dB`
- **Wet-Tail**: `-26.24 dB`（v5 の乾燥 `-29.50 dB` から接着回復）
- **ハモリ処理**: モノラルハモリ 1 本を対称 `11 ms` Haas で自然なステレオ展開＋芯抜き／語頭潰し適用
- **Studio One 5.5 監査**: XML 正常 / ネイティブ `.vstpreset` 17 個 / JUCE リーク `0`

---

## 3. 新規配置済み未検証素材

- `C:\MIX_automation\data\raw\エバ`（`2026-11-26` にユーザー配置済み、未実行）
