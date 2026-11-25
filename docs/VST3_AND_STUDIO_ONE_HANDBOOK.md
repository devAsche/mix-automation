# VST3 プラグイン実機仕様 & Studio One 5.5 連携ハンドブック

## 1. 実機検証済み 27 VST3 プラグイン一覧（`plugin_registry.py`）

配置ルート: `C:\Program Files\Common Files\VST3`（ディレクトリバンドル型 `.vst3` は `PluginSpec.loadable_path` により `Contents\x86_64-win\<name>.vst3` を解決してロードする）。

| Key | Plugin Name | Role | 備考・実機知見 |
| :--- | :--- | :--- | :--- |
| `rx10_mouth_declick` | iZotope RX 10 Mouth De-click | `mouth_declick` | オフライン下処理専用 |
| `rx10_deplosive` | iZotope RX 10 De-plosive | `deplosive` | オフライン下処理専用 |
| `rx10_voice_denoise` | iZotope RX 10 Voice De-noise | `voice_denoise` | オフライン下処理専用 |
| `rx10_breath_control` | iZotope RX 10 Breath Control | `breath_control` | オフライン下処理専用 |
| `rx10_deess` | iZotope RX 10 De-ess | `deesser` | Spectral モードでアタック保護 |
| `persistent_q` | Persistent Q | `subtractive_eq` | **873 パラメータ**。v6 では 16 バンド（Static+Notch+Dynamic+Spectral）を運用 |
| `zl_spectrum_eq` | ZL Spectrum Equalizer | `subtractive_eq` | 代替／補助マルチバンド EQ |
| `vca_compressor` | Softube VCA Compressor | `peak_compressor` | **`attack` (`0.25..4.0`) は小さいほど高速**。リード=`0.52–0.72, drive=0.1`、ハモリ語頭潰し=`0.25, rel=3.40, 6:1` |
| `uad_la2a` | UAD Teletronix LA-2A Tube Compressor | `optical_compressor` | `peak_reduct`（GR `2.0–3.6 dB`）と `gain`（ユニティ RMS）の**二重二分探索が必須** |
| `uad_pultec_eqp1a` | UAD Pultec EQP-1A | `color_eq` | Pop/Ballad カラー EQ |
| `uad_pultec_meq5` | UAD Pultec MEQ-5 | `color_eq` | 中域彫刻カラー EQ |
| `slick_eq` | TDR VOS SlickEQ | `color_eq` | 音楽的 3 バンドカラー EQ |
| `izotope_vea` | iZotope VEA | `vocal_enhancer` | Clean/Shape/Boost |
| `soothe2` | Oeksound Soothe2 | `resonance_suppressor` | **周波数パラメータは `'2k5'`, `'6k5'`, `'10k'` の kHz 文字列トークン**（`_KHZ_RE` で処理）。探索中 `eco` / 最終 `high` |
| `saturation_knob` | Softube Saturation Knob | `saturator` | 必ず `240 Hz–4.0 kHz` 中域限定バンドパス内でパラレル適用（`KEEP LOW`） |
| `izotope_trash2` | iZotope Trash 2 | `saturator` | 歪み・フィルター特殊効果 |
| `initial_dynamic_eq` | Initial Dynamic EQ | `dynamic_unmask_eq` | 動的アンマスキング補助 |
| `rev_plate_140` | Arturia Rev PLATE-140 | `plate_reverb` | プレートリバーブ |
| `exponential_r2` | Exponential Audio R2 | `space_reverb` | `mix` は `"Wet"` / `"Dry"` 文字列エイリアス対応 |
| `phoenix_verb` | Exponential Audio PhoenixVerb | `space_reverb` | 自然なルーム・ホール・儚いテール。`mix` は `"Wet"` / `"Dry"` 対応 |
| `khs_dual_delay` | Kilohearts Dual Delay | `delay` | ステレオ／ピンポンディレイ |
| `vocal_doubler` | iZotope Vocal Doubler | `doubler` | ダブル幅拡張 |
| `mautopitch` | MeldaProduction MAutoPitch | `pitch_fx` | ケロケロ／フォルマントシフト |
| `izotope_nectar3` | iZotope Nectar 3 | `pitch_fx` | オフライン専用。コード整合仮歌ハモリ生成 |
| `izotope_vocalsynth2` | iZotope VocalSynth 2 | `pitch_fx` | **317 パラメータ（ライセンス取得済み）**。純 FX（ラジオ等）は `input_lvl_dry_corrected=100.0`、Biovox は `bvx_enable=True, bvx_lvl=100.0` が必須（未設定だと無音出力） |
| `ozone9_elements` | iZotope Ozone 9 Elements | `master_processor` | **120 パラメータ露出**（`Ozone 12 Elements` は Pedalboard 上で 0 パラメータのため 9 を採用） |
| `wave_breaker` | Press Play Wave Breaker | `true_peak_limiter` | トゥルーピーク・リミッター |

- **除外済み**: `VocAlign 6 Standard`（ライセンス期限切れのためレジストリから除外）。

---

## 2. Studio One 5.5 (`5.5.2.86528`) ネイティブ `.song` 連携仕様

Studio One 5.5 は `.dawproject` 非対応のため、ベーステンプレート（`data/templates/test20261124/test20261124.song`）をベースに ZIP 内の XML・Media・Presets を直接合成する。

### 2.1 JUCE Base64 → ネイティブ Steinberg `.vstpreset` 復号

Pedalboard が返す `plugin.raw_state` は JUCE ラッパー形式：
```xml
<VST3PluginState>
  <IComponent>{num_bytes}.{JUCE独自Base64}</IComponent>
  <IEditController>{num_bytes}.{JUCE独自Base64}</IEditController>
</VST3PluginState>
```
JUCE 独自 Base64 アルファベットは `.ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+` であり、標準 Base64 ではない。`song_builder.py` の `decode_juce_base64()` と `extract_steinberg_comp_and_cont()` によりネイティブ Steinberg VST3 の `Comp` / `Cont` チャンクへ復号して `.vstpreset` へ格納することで、Studio One 5.5 で全ノブが 1:1 再現される（`<VST3PluginState>` リークは常に `0` 件であることを自動検証する）。

### 2.2 検証済み Studio One 5.5 ClassID テーブル（`DEFAULT_S1_PLUGIN_SPECS`）

`%APPDATA%\PreSonus\Studio One 5\x64\Plugins-*.settings` と照合済みの主要クラス ID：

| Key | Studio One Display Name | ClassID | SubCategory |
| :--- | :--- | :--- | :--- |
| `persistent_q` | `Persistent Q` | `{ABCDEF01-9182-FAEB-4B61-706550713031}` | `VST3` |
| `vca_compressor` | `VCA Compressor` | `{24DCAB96-A33C-41E5-B2C9-7A9953B51CCD}` | `VST3/Dynamics` |
| `uad_la2a` | `UADx LA-2A Tube Compressor` | `{ABCDEF01-9182-FAEB-5541-44785533444A}` | `VST3/Dynamics` |
| `uad_pultec` | `UADx Pultec EQP-1A EQ` | `{ABCDEF01-9182-FAEB-5541-44785533414E}` | `VST3/EQ` |
| `slick_eq` | `TDR VOS SlickEQ` | `{56535454-6431-3074-6472-20766F732073}` | `VST3/EQ` |
| `soothe2` | `soothe2` | `{ABCDEF01-9182-FAEB-4F65-6B5353746842}` | `VST3/Dynamics` |
| `vocalsynth2` | `VocalSynth 2` | `{56535456-7853-3256-6F63-616C53796E74}` | `VST3/Mixing` |
| `saturation_knob` | `Saturation Knob` | `{F73F44A6-96F4-40AF-B187-1EAFA5E2AF01}` | `VST3/Distortion` |
| `phoenix_verb` | `PhoenixVerb` | `{5653544D-4373-7270-686F-656E69787665}` | `VST3/Reverb` |
| `r2` | `R2` | `{5653544D-4373-3272-3200-000000000000}` | `VST3/Reverb` |
| `wave_breaker` | `Wave Breaker` | `{F2AEE70D-00DE-4F4E-5050-6C7950507762}` | `VST3/Dynamics` |
| `h_delay` | `H-Delay Stereo` | `{56535448-4244-5368-2D64-656C61792073}` | `VST3/Delay` |
| `doubler2` | `Doubler2 Stereo` | `{56535457-4432-5364-6F75-626C65723220}` | `VST3/Modulation` |
| `mv2_mono` | `MV2 Mono` | `{5653544D-5632-4D6D-7632-206D6F6E6F00}` | `VST3/Dynamics`（テンプレート互換用・v6 本線では不使用） |

### 2.3 契約テストによる保護

`tests/test_vst3_contracts.py` が実機プラグインをロードし、離散パラメータ強制・`Soothe2` kHz パース・16 バンド `Persistent Q`・`VocalSynth 2` 有音出力・JUCE Base64 復号の契約を自動保証する（全 21 テスト：`pytest tests -v`）。
