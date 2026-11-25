# VST3 プラグイン実機仕様 & Studio One 5.5 連携ハンドブック (`v7.0`)

## 1. 実機検証済み 27 VST3 プラグイン一覧（`plugin_registry.py`）

配置ルート: `C:\Program Files\Common Files\VST3`（ディレクトリバンドル型 `.vst3` は `PluginSpec.loadable_path` により `Contents\x86_64-win\<name>.vst3` を解決してロードする）。

| Key | Plugin Name | Role | 備考・実機知見 |
| :--- | :--- | :--- | :--- |
| `rx10_mouth_declick` | iZotope RX 10 Mouth De-click | `mouth_declick` | オフライン下処理専用 |
| `rx10_deplosive` | iZotope RX 10 De-plosive | `deplosive` | オフライン下処理専用 |
| `rx10_voice_denoise` | iZotope RX 10 Voice De-noise | `voice_denoise` | `'Music'` + `'Gentle'` モード厳守 |
| `rx10_breath_control` | iZotope RX 10 Breath Control | `breath_control` | オフライン下処理専用 |
| `rx10_deess` | iZotope RX 10 De-ess | `deesser` | `'Spectral De-ess'` (`> 5.8 kHz`) でアタック保護 |
| `persistent_q` | Persistent Q | `subtractive_eq` | **873 パラメータ**。16 バンド（Static+Notch+Dynamic+Spectral）運用 |
| `zl_spectrum_eq` | ZL Spectrum Equalizer | `subtractive_eq` | 代替／補助マルチバンド EQ |
| `vca_compressor` | Softube VCA Compressor | `peak_compressor` | **`attack` (`0.25..4.0`) は小さいほど高速**。リード=`0.52–0.72, drive=0.1`、ハモリ語頭潰し=`0.25, rel=3.40, 6:1` |
| `uad_la2a` | UAD Teletronix LA-2A Tube Compressor | `optical_compressor` | `peak_reduct` と `gain` の**二重二分探索が必須** |
| `uad_pultec_eqp1a` | UAD Pultec EQP-1A | `color_eq` | Pop カラー EQ（`hf_boost=0.0` で高域過多防止） |
| `uad_pultec_meq5` | UAD Pultec MEQ-5 | `color_eq` | 中域彫刻カラー EQ（10 パラメータ、S1 ClassID 登録済み） |
| `slick_eq` | TDR VOS SlickEQ | `color_eq` | Ballad カラー EQ（`Silky`） |
| `izotope_vea` | iZotope VEA | `vocal_enhancer` | Clean/Shape/Boost |
| `soothe2` | Oeksound Soothe2 | `resonance_suppressor` | **周波数は `'2k5'`, `'6k5'`, `'10k'` トークン**。探索中 `eco` / 最終 `high` |
| `saturation_knob` | Softube Saturation Knob | `saturator` | `240 Hz–4.0 kHz` 中域限定バンドパス内でパラレル適用（`KEEP LOW`） |
| `izotope_trash2` | iZotope Trash 2 | `saturator` | **505 パラメータ**。v7 Stage ③ `distortion_grit` で統合（`trash_stg1_filter_enable=True`, `380–4200 Hz` 制限） |
| `initial_dynamic_eq` | Initial Dynamic EQ | `dynamic_unmask_eq` | 動的アンマスキング補助（S1 ClassID 登録済み） |
| `rev_plate_140` | Arturia Rev PLATE-140 | `plate_reverb` | プレートリバーブ（S1 ClassID 登録済み） |
| `exponential_r2` | Exponential Audio R2 | `space_reverb` | `mix` は `"Wet"` / `"Dry"` 文字列エイリアス対応 |
| `phoenix_verb` | Exponential Audio PhoenixVerb | `space_reverb` | 自然なルーム・儚いスローテール |
| `khs_dual_delay` | Kilohearts Dual Delay | `delay` | **13 パラメータ**。v7 Stage ③ `spot_delay_throw` で統合。文字列に Unicode `\u2212` (`−`)・`\u202f` を含むため `vst3_adapter.py` で正規化済み |
| `vocal_doubler` | iZotope Vocal Doubler | `doubler` | ダブル幅拡張（5 パラメータ、S1 ClassID 登録済み） |
| `mautopitch` | MeldaProduction MAutoPitch | `pitch_fx` | **29 パラメータ**。v7 Stage ③ `hard_autotune` で統合（`c..b` 12 音スケール bool 対応、`bypass` は `['Off']` のみのため `False` 代入禁止） |
| `izotope_nectar3` | iZotope Nectar 3 | `pitch_fx` | オフライン専用。コード整合仮歌ハモリ生成 |
| `izotope_vocalsynth2` | iZotope VocalSynth 2 | `pitch_fx` | **317 パラメータ（ライセンス済み）**。純 FX は `input_lvl_dry_corrected=100.0`、Biovox は `bvx_enable=True, bvx_lvl=100.0` が必須 |
| `ozone9_elements` | iZotope Ozone 9 Elements | `master_processor` | **120 パラメータ露出**（`Ozone 12 Elements` は 0 パラメータのため 9 を採用、S1 ClassID 登録済み） |
| `wave_breaker` | Press Play Wave Breaker | `true_peak_limiter` | トゥルーピーク・リミッター |

- **除外済み**: `VocAlign 6 Standard`（ライセンス期限切れのためレジストリから除外）。

---

## 2. Studio One 5.5 (`5.5.2.86528`) ネイティブ `.song` 連携仕様

### 2.1 JUCE Base64 → ネイティブ Steinberg `.vstpreset` 復号 & スポット FX トラック
Pedalboard の `plugin.raw_state`（JUCE 独自 Base64 `.A-Za-z0-9+` の `<VST3PluginState>`）を `song_builder.py` の `extract_steinberg_comp_and_cont()` でネイティブ Steinberg `Comp`/`Cont` へ復号する（`<VST3PluginState>` リーク `0` 件を契約テストと `benchmark-ab` の双方で自動保証）。
さらに v7.0 では `TrackMediaSpec.insert_slots` と `export_stage3_fx_vst3_states()` により、`fx_radio_vocalsynth2`・`fx_autotune_mautopitch`・`fx_distortion_trash2`・`fx_spot_delay` の各スポットトラックへ本物の VST3 インサートと `.vstpreset` を直接配置し、全追加トラックに正規の空 `.envelopex`（`ボリューム.envelopex` `b"{i\x06events[]}"`, `パン.envelopex` `b"{i\x07bipolari\x01i\x06events[]}"`）と細分化 `AudioEvent` を付与する。

### 2.2 検証済み Studio One 5.5 ClassID テーブル（`DEFAULT_S1_PLUGIN_SPECS` 全 22 種）

| Key | Studio One Display Name | ClassID | SubCategory |
| :--- | :--- | :--- | :--- |
| `persistent_q` | `Persistent Q` | `{ABCDEF01-9182-FAEB-4B61-706550713031}` | `VST3` |
| `initial_dynamic_eq` | `Initial Dynamic EQ` | `{ABCDEF01-9182-FAEB-4961-756464596E33}` | `VST3` |
| `vca_compressor` | `VCA Compressor` | `{24DCAB96-A33C-41E5-B2C9-7A9953B51CCD}` | `VST3/Dynamics` |
| `uad_la2a` | `UADx LA-2A Tube Compressor` | `{ABCDEF01-9182-FAEB-5541-44785533444A}` | `VST3/Dynamics` |
| `uad_pultec` | `UADx Pultec EQP-1A EQ` | `{ABCDEF01-9182-FAEB-5541-44785533414E}` | `VST3/EQ` |
| `uad_pultec_meq5` | `UADx Pultec MEQ-5 EQ` | `{ABCDEF01-9182-FAEB-5541-447855334152}` | `VST3/EQ` |
| `slick_eq` | `TDR VOS SlickEQ` | `{56535454-6431-3074-6472-20766F732073}` | `VST3/EQ` |
| `soothe2` | `soothe2` | `{ABCDEF01-9182-FAEB-4F65-6B5353746842}` | `VST3/Dynamics` |
| `vocalsynth2` | `VocalSynth 2` | `{56535456-7853-3256-6F63-616C53796E74}` | `VST3/Mixing` |
| `mautopitch` | `MAutoPitch` | `{4D656C64-6170-726F-4D61-75544D617554}` | `VST3/Pitch & Time` |
| `izotope_trash2` | `iZotope Trash 2` | `{5653545A-6E54-3269-5A6F-746F70652054}` | `VST3/Distortion` |
| `khs_dual_delay` | `kHs Dual Delay` | `{00ED3532-00A9-4557-0028-037A00175A0E}` | `VST3/Spatial` |
| `vocal_doubler` | `Vocal Doubler` | `{56535449-5A56-4456-6F63-616C20446F75}` | `VST3/Modulation` |
| `rev_plate_140` | `Rev PLATE-140` | `{41727475-4156-4953-5265-506C50726F63}` | `VST3/Reverb` |
| `saturation_knob` | `Saturation Knob` | `{F73F44A6-96F4-40AF-B187-1EAFA5E2AF01}` | `VST3/Distortion` |
| `phoenix_verb` | `PhoenixVerb` | `{5653544D-4373-7270-686F-656E69787665}` | `VST3/Reverb` |
| `r2` | `R2` | `{5653544D-4373-3272-3200-000000000000}` | `VST3/Reverb` |
| `ozone9_elements` | `Ozone 9 Elements` | `{5653545A-6E45-394F-7A6F-6E6520392045}` | `VST3/Mastering` |
| `wave_breaker` | `Wave Breaker` | `{F2AEE70D-00DE-4F4E-5050-6C7950507762}` | `VST3/Dynamics` |
| `h_delay` | `H-Delay Stereo` | `{56535448-4244-5368-2D64-656C61792073}` | `VST3/Delay` |
| `doubler2` | `Doubler2 Stereo` | `{56535457-4432-5364-6F75-626C65723220}` | `VST3/Modulation` |
| `mv2_mono` | `MV2 Mono` | `{5653544D-5632-4D6D-7632-206D6F6E6F00}` | `VST3/Dynamics`（テンプレート互換用・本線不使用） |

### 2.3 契約テストによる保護
`tests/test_vst3_contracts.py`（7 契約テスト）＋ `tests/test_core.py`（18 単体テスト）の計 25 テストが、全商用 VST3 の有音応答・周波数特性・Unicode トークン解決・JUCE リーク 0 件を自動保証する。
