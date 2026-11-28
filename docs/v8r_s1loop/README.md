# R5: 補正 EQ ループの一周目（2026-11-28、Claude 作業）

方針（凌さんと合意、`docs/evaluations/2026-11-28_R4聴き比べ.txt`）:
- プロのカバーは「オケとの関係」（声−オケの音量差、空間、広がり、ダイナミクス、オケのダッキング）だけに使う。
- 音色は凌さんの耳で決める。リードの最後に Persistent Q を 1 つ足し、凌さんは Studio One でそれだけを触る。
- 保存された設定を読み取り、数曲分をまとめて「声の補正 EQ」として固定し、全曲の最後に掛ける（平均は 2 回目以降）。

## 0. プラグイン

凌さんの「Persistent EQ」は `C:\Program Files\Common Files\VST3\Persistent Q.vst3`（plugin_registry の `persistent_q`）。v8 前段の引き算 EQ と同じ VST3 で、この名前の製品は 1 つだけ。

## 1. 目的関数（`--r5`）

コード: `knobs.R5_TONE`、`match.STAGE_A_R5`・`WEIGHTS_R5`・`SongMatch(r5=True)`、`scripts/match/run_match_fast.py --r5`。R4 / R4b の定義はそのまま残した。

### 音色の固定値（探索しない）

| つまみ | R3 既定 | R4 最適（エバ / L&D） | **R5 固定** | 根拠 |
|---|---|---|---|---|
| hpf_hz | 140 | – | **120** | Mo_qfyNdYQQ@2:38（119 Hz）。140 は「芯が細い」に対して高め |
| 500 Hz | −1.5 dB | 同じ | −1.5 dB（チェーン固定） | 2afvdjOpVJQ@4:17「少し」 |
| presence_db | 0 | +0.37 / +1.14 | **0（使わない）** | 3k のシューシュー音が安っぽさの原因 RTVPdDmWrwo@34:37 |
| Pultec brightness / air | 2.5 / 8 kHz | 4.46 / 5.75、10 kHz | **1.5 / 10 kHz** | R4 の 4.5〜5.8 で「シャリつく」。R3 より 1 下げ、足りなければ補正 EQ で足す。10 kHz は articles [8] |
| Pultec 低域 | 100 Hz boost 2 / atten 3 | 同じ | 同じ（チェーン固定） | -c0bHeO5ISs@19:22 |
| saturation | 1.5 | 0.31 / 0.02 | **1.5** | 2afvdjOpVJQ@8:02（軽く）。R4 の 0 付近は「芯が細い」側 |
| deess_db | 3.0 | 6.89 / 3.86 | **3.0** | RTVPdDmWrwo@26:38（1 台で大きく下げない） |

### 探索するもの・距離

- 段 A（ダイナミクス）: comp_total_gr_db, pre_level_strength, par_comp_mix。
- 段 B（音量・空間・広がり・ダッキング）: R4b の STAGE_B のまま（vocal_offset_lu, ride_tolerance_lu, センド 3 本, plate_decay_s, delay_send_db, fx_duck_db, inst_duck_max_db, inst_side_high_db）。
- 重み: 声−オケのスペクトル形 `spec` と AFx-Rep `afx` を 0（コードは残す）。他は R4b と同じ（lu 2、sm・tail 1、dyn・crest・inst_spec・inst_sm・duck 0.5）。R2 の決定（V−I はメインだけの区間、ダッキングの帯域 300 Hz〜6 kHz、尾は片側）はそのまま。
- 比較用に、各評価で R4b の重みの距離 `dist_r4b` も残す。

（結果は下の 6 章）

## 2. 補正 EQ の段

- 位置: リードの最後。kHs Limiter → phrase ride（純粋なゲイン）→ **補正 EQ（Persistent Q）**。
- v1 は乾いた声だけに掛ける。ダブラー・ルーム・プレート・ディレイへの送りと、オケのダッキングのキーは、補正 EQ の**前**の声から取る。
- 設定: `profiles/correction_eq/<声>.json`（既定の声は `ryo`）。無い、または有効な帯域が無いときは段を飛ばす（プラグインも読み込まない）。
- `mix-auto mix … --engine v8 --voice ryo` / `--correction-eq <json>`。

## 3. Studio One 書き出し

コード: `src/mix_automation/v8/s1_export.py`、`src/mix_automation/v8/s1_preset.py`（.vstpreset ↔ pedalboard raw_state）、`scripts/s1/export_song.py <v8 出力フォルダ>`。テンプレートは `data/templates/test20261124/test20261124.song`（読むだけ）。

| トラック | 中身 | プラグイン |
|---|---|---|
| Vocal CorrEQ | リード（ride の後・補正 EQ の前）`stems/lead_dry.wav` | **Persistent Q（全帯域無効＝素通し）← 凌さんが触るのはここだけ** |
| 空間 4 本 | wet_doubler / wet_room / wet_plate / wet_delay | なし |
| Inst | ダッキング後のオケ | なし |
| Ref v8 (pre-EQ)（ミュート） | 上の全トラックの和（補正 EQ が平坦のとき） | なし |

- 全 WAV（float、`s1_export/s1_media/`）に、パイプラインのマスター入力ゲイン（mix_work_gain）を先に掛けた。フェーダーは全部 0 dB。
- マスター: Saturation Knob（KEEP LOW 0.8）→ VCA Compressor 並列グルー → Wave Breaker（threshold −0.8、drive_db をリミッター前のゲインに使う）。**M/S コンプ（Mid、平均 GR 1.5 dB）は scipy の処理で実機プラグインが無いので省いた。**
- 参照トラックに完成版の Master.wav を入れると、マスターの処理が 2 回かかる。そこで「マスターを通すと完成版になる、マスター前の和」を入れた。比べるときは、他のトラックをミュートして、これだけを鳴らす。完成版そのものは `<曲>_v8/<曲>_v8_Master.wav`。
- 元の .song を `<曲>_v8r5_orig.song` として別名でも残した。

### マスター再現（書き出した preset を pedalboard で読み戻して通し、完成版と比べた）

| 曲 | 経路 | LUFS 差 | 完成版とのヌル残差 |
|---|---|---|---|
| エバ | プラグイン＋M/S コンプ（確認用） | +0.005 LU | −62.5 dB |
| エバ | **.song の経路（M/S コンプなし）** | **+0.092 LU** | −20.5 dB |
| L&D | プラグイン＋M/S コンプ（確認用） | −0.008 LU | −60.6 dB |
| L&D | **.song の経路（M/S コンプなし）** | **+0.090 LU** | −20.4 dB |

- M/S コンプを省くと、音量が約 0.9 LU 上がる。Wave Breaker の drive を合わせ直して、完成版との差を 0.1 LU 以内にした（エバ 7.03 → 6.33 dB、L&D 5.97 → 5.13 dB）。
- 残差 −20 dB は M/S コンプ（Mid だけ 1.5 dB 圧縮）を省いた分。補正 EQ の判断（声の音色）への影響は小さいとみている。
- 開けるかの確認（GUI は使っていない）: zip と XML の構文、メディアの実在、各 preset の読み戻し、ID の重複なし → OK。**Studio One で実際に開いたことはまだ無い。**

## 4. 読み取りと適用

- `scripts/s1/read_correction_eq.py <song> [--track "Vocal CorrEQ"] [--voice ryo]`: .song の中の `Presets/Channels/<チャンネル>/<n> - Persistent Q.vstpreset` を、`Devices/audiomixer.xml` と突き合わせて特定する。中身を pedalboard の raw_state に戻して Persistent Q に読み込み、帯域の値を JSON にする（`<PersistantQState>` の XML とも突き合わせる）。
- 保存先は `profiles/correction_eq/<声>.json`。既にあれば上書きせず、`<声>_<時刻>.json` に書く。本番に使うファイルへの切り替えは人が決める。
- 確認:
  - Studio One が自動保存した既存の .song（エバ v7、mainvocal）: 16 帯域を読めた。XML と一致。
  - 今回書き出した .song: 平坦（有効な帯域 0）。
- 往復テスト（`tests/test_v8_r5_s1.py`）: 200 Hz Low Shelf −1.5 / 3.2 kHz Peaking −2 Q 1.4 / 10 kHz High Shelf +1 を .song に書いて読み戻した。
  - 値の一致: −2.0 → −1.98 dB（プラグインの刻み）。
  - 読み戻した設定で掛けた音と元の設定で掛けた音のヌル残差: −300 dB（ビット一致）。
- 平坦のとき: 段を飛ばすので、補正段が無い場合と出力はビット一致（テスト）。
- 平均: `correction_eq.average_profiles`。各設定の dB 応答を 1/24 オクターブの log 周波数格子で平均し、帯域に当てはめ直す。応答は RBJ の解析式で、Persistent Q 実機との差は最大 0.02 dB。2 回目以降に使う。

## 5. 言葉 → つまみ・帯域の対応（凌さんが EQ 以外を言葉で言ったとき）

| 言葉 | まず当てる所 | 次の候補 | 根拠 |
|---|---|---|---|
| 刺さる・シャリつく・痛い | 補正 EQ: 3〜5 kHz を Peaking で −1〜−2 dB / Pultec brightness を下げる | deess_db を上げる（歯擦音だけなら） | 3k のシューシュー RTVPdDmWrwo@34:37、歯擦音はディエッサー heG8RtaRwN0@6:21 |
| こもる・抜けない | 補正 EQ: 250〜500 Hz を −1 dB、または 1.7〜2 kHz の溜まりを確認 | 高域を足す前に低中域を削る | 500 Hz 2afvdjOpVJQ@4:17、1.7〜2k RTVPdDmWrwo@27:49 |
| 芯が細い・薄い | 補正 EQ: 150〜300 Hz を Low Shelf / Peaking で +1 dB、HPF を下げる | saturation を上げる、1.5 kHz（芯）を少し | 倍音側 1.5 kHz E8zUKz21S9c@21:21、250 Hz 温かみ 2afvdjOpVJQ@3:28 |
| 浮く | vocal_offset_lu を下げる | オケ側 1〜5 kHz のダッキングを浅く（下げすぎると浮く）、センドを少し上げる | 知恵袋 q10188896595、浮く・埋もれるはダイナミクス 4-GZf4N1cso@4:10 |
| 埋もれる | vocal_offset_lu を上げる | comp_total_gr_db を上げる、inst_duck_max_db | 4-GZf4N1cso@4:10、-c0bHeO5ISs@3:47 |
| 硬い・つぶれている | comp_total_gr_db を下げる | par_comp_mix を下げる | 2.4 コンプ（軽めの多段） |
| 響きすぎ・遠い | センド（plate / room）を下げる、fx_duck_db を上げる | plate_decay_s を短く | 空間は「入れると分からない量」JUXJShxfzKQ@14:30 |
| 乾いている・近すぎる | room / plate のセンドを上げる | doubler を上げる | JUXJShxfzKQ@8:49 |

## 6. 結果（2026-11-28）

### 探し直し（各 15 分予算、6 並列）

| 曲 | 設定 | dist（R5） | dist_r4b | 分 | 評価 A / B |
|---|---|---|---|---|---|
| エバ | R4 最適（音色込み、比較用） | 0.499 | 0.589 | （R4: 120） | – |
| エバ | R4 最適 + R5 音色 | 1.113 | 1.094 | – | – |
| エバ | **R5 最終** | **0.755** | 0.863 | 14.5 | 48 / 72 |
| L&D | R4 最適（音色込み、比較用） | 0.711 | 0.939 | （R4: 120） | – |
| L&D | R4 最適 + R5 音色 | 1.663 | 1.567 | – | – |
| L&D | 1 回目（エバの R5 から開始） | 1.663（探索は 2.11 止まり） | 1.567 | 11.6 | 84 / 48 |
| L&D | **R5 最終（R4 最適 + 音色から開始）** | **1.515** | 1.479 | 6.5 | 60 / 36 |

- 音色を固定した結果、距離は R4 最適より大きい。主な原因は `crest`（声のピーク − 活動 RMS）。エバ Ivudotto の場合、R4 最適で 0.01、R5 音色で 6.4 だった。crest はサチュ・Pultec の設定と強く結びついていて、音量と空間のつまみでは戻せない。もともと「プロ側はマスター後なので小さく出る」偏りがある項（R4 README の注意）。
- 一方、声−オケのスペクトル形（重み 0 にした spec）は、R5 の穏やかな音色の方がプロに近かった（エバ 0.83 → 0.50、L&D 1.29 → 0.95）。
- つまみ（R5 最終）:

| つまみ | エバ | L&D |
|---|---|---|
| vocal_offset_lu | +1.56 | −1.37 |
| comp_total_gr_db | 6.90 | 9.34 |
| pre_level / par_comp | 0.36 / 0.07 | 0.61 / 0.20 |
| doubler / room / plate send dB | −14.3 / −12.1 / −20.2 | −14.1 / −12.1 / −15.9 |
| delay_send_db | −24.4 | −17.2 |
| fx_duck_db / inst_duck_max_db | 1.13 / 0.84 | 1.83 / 1.34 |

### 全曲レンダー（補正 EQ は平坦）

`data/outputs_v8r5/render/<曲>_v8/`。knobs は `data/outputs_v8r5/knobs/エバ.json`、`ラブ&デストロイ_20261128-170555.json`。

| 曲 | 声−オケ LU | 声 S/M dB | マスター LUFS |
|---|---|---|---|
| エバ | +2.16 | −13.9 | −9.58 |
| L&D | −0.76 | −14.0 | −9.55 |

### 凌さんが開く .song

- エバ: `data\outputs_v8r5\render\エバ_v8\s1_export\エバ_v8r5.song`
- L&D: `data\outputs_v8r5\render\ラブandデストロイ_v8\s1_export\ラブandデストロイ_v8r5.song`

手順: 開く → 「Vocal CorrEQ」トラックの Persistent Q だけを回す → 保存（上書きしてよい。元は `_orig.song` に残っている）。そのあと `scripts/s1/read_correction_eq.py <song>` で読み取る。

## 判断が必要な点

1. crest 項: 音色を固定すると、距離の大半を crest が占める。マスター後のプロとは比べにくい項なので、R5 の重みを 0 にするか、マスター後の v8 で測る。
2. Studio One で実際に開けるかは、凌さんが開くまで分からない。開けなければ、エラーの内容から直す。
3. 補正 EQ の位置は v1 では乾いた声だけ。リバーブ等の返りにも同じ EQ を掛けるか（v2）は、凌さんの最初の設定を見てから決める。
