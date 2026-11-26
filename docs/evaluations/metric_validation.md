# 指標の検証（Phase 2）

`scripts/analysis/evaluate_issues.py --write-md` の出力。指標の定義と出典は `src/mix_automation/v8/metrics/`、参照分布は `reference_stats.json`（`scripts/analysis/build_reference_stats.py`）。
参照: プロ 87 曲（BS-RoFormer 分離済みステム）、自MIX 2 件。

## 指摘 × 指標

判定: HIT(abs) = 絶対基準（プロ分布・自MIX・JND）を超えた、HIT(rel) = 同じ曲の残りの時間の 10/90 パーセンタイルを超えた。時刻窓は ±1 s（approx は ±3 s）、「～／から」は終わりを +3 s。閾値は issues.yaml からは決めていない。「同程度」= 同じ曲で同じくらい悪い箇所の数（指摘されていない箇所は誤報に数えない）。

| id | 分類/向き | 時刻 | 指標 | 値 | 絶対閾値 | 根拠 | 曲内閾値 | 判定 | 同程度 | 補足 |
|---|---|---|---|---|---|---|---|---|---|---|
| fil-g1 | airy_fizzy/too_airy | 全体 | R_airy: air_rel v7−本家 (dB) | — | — | — | — | — | — | 本家音源なし |
| fil-g1 | airy_fizzy/too_airy | 全体 | R_airy_abs: ボーカル自身 8-16k−1-4k v7−本家 (dB) | — | — | — | — | — | — | 本家音源なし |
| fil-1 | consonant_attack/too_strong | 2:34.0 | C attack Δ子音/母音(ドライ, dB) | 4.82 | 7.66 | 比の絶対値 vs プロ p95 | 3.73 | HIT(rel) | 22/422 個 | @2:33.3 比の絶対値 -23.3 dB, V/I -48.5 dB, Δcrest 5.3 |
| eva-g1 | space_excess/too_much | 全体 | R_space_tail: フレーズ後の残り v7−本家 (dB) | -14.04 | 1.00 | JND（v7 分離版−本家 分離） | — | miss | — | v7 ステム−本家 0.43; ドライ再レンダー−本家 — |
| eva-g1 | space_excess/too_much | 全体 | R_space_decay: 減衰時間 v7/本家−1 | -0.04 | 0.05 | JND（v7 分離版−本家 分離） | — | miss | — | v7 ステム−本家 0.17; ドライ再レンダー−本家 — |
| eva-g2 | buried/buried | 全体 | R_buried: 部分ラウドネス損失 v7−本家 (phon) | 4.13 | -1.00 | JND（v7 分離版−本家 分離） | — | miss | — | v7 ステム−本家 2.94; ドライ再レンダー−本家 — |
| eva-g3 | thin_light/too_light | 全体 | R_thin: body_rel v7−本家 (dB) | 6.89 | -1.00 | JND（v7 分離版−本家 分離） | — | miss | — | v7 ステム−本家 6.56; ドライ再レンダー−本家 — |
| eva-1 | sibilance/crushed | 0:20.0 | C sibilant Δ子音/母音(ドライ, dB) | 0.58 | — | （量ではなく痕跡で判定: Q 行） | 1.17 | HIT(rel) | 5/61 個 | [Whisper で特定できず: 時刻窓で最悪の子音] @0:21.0 比の絶対値 -15.2 dB, V/I 5.4 dB, Δcrest 1.8 |
| eva-1 | sibilance/crushed | 0:20.0 | Q ディエッサー痕跡 raw 基準: (a)帯域減衰 dB | -9.77 | -3.00 | 1/3 oct の谷の可聴 ~3 dB (Bücklein 1981) | -2.13 | HIT(abs+rel) | 1/61 個 | [Whisper で特定できず: 時刻窓で最悪の子音] @0:21.0 ピーク 3919 Hz, (b)重心低下 12.2% (曲内 p90 5.3), (c)ゲインの谷 -8.4 dB (曲内 p10 -4.8), 残差 3.8 dB, LSD 2.8 dB; うち下処理(prepared−raw) 帯域 -9.9 / 谷 -7.4 dB |
| eva-1 | sibilance/crushed | 0:20.0 | Q ディエッサー痕跡 prepared 基準: (a)帯域減衰 dB | 0.04 | -3.00 | 1/3 oct の谷の可聴 ~3 dB (Bücklein 1981) | -0.04 | miss | 8/61 個 | [Whisper で特定できず: 時刻窓で最悪の子音] @0:21.0 ピーク 3833 Hz, (b)重心低下 -3.2% (曲内 p90 2.6), (c)ゲインの谷 -2.2 dB (曲内 p10 -2.3), 残差 6.2 dB, LSD 0.9 dB |
| eva-2 | buried/buried | 1:31.0 | A 部分ラウドネス損失(phon, 窓内 0.4 s ブロック中央値) | -10.17 | -20.23 | プロ全ブロック p5 | -14.64 | miss | 172/355 ブロック | 1-4k V/I 1.9 dB |
| eva-2 | buried/buried | 1:31.0 | A ドライのリード vs オケ+ウェット+ハモリ 損失(phon) | -9.94 | — | — | -15.09 | miss | 267/355 ブロック |  |
| eva-2 | buried/buried | 1:31.0 | I ESTOI（処理前→マスター、フレーズ） | 0.59 | 0.06 | 本家（分離ボーカル→原曲）p10 | 0.59 | miss | 2/11 フレーズ |  |
| eva-2 | buried/buried | 1:31.0 | I ASR 文字誤り率（処理前の書き起こし基準、マスター） | 0.09 | 1.50 | 本家（分離ボーカル→原曲）p90 | 0.84 | miss | 8/11 チャンク | 処理前「心の対から腐ってる空想由来につき」/ マスター「真の対から腐ってる空想由来につき」 |
| eva-3 | muddy/too_muddy | 2:01.0 | B bright_rel (6 s 窓) | 0.60 | -1.00 | 以降−以前 の差 ≤ −1 dB(JND) | -3.75 | miss | 116/164 窓 | 以降−以前 -0.98 dB |
| eva-4 | buried/buried | 2:42.0 | A 部分ラウドネス損失(phon, 窓内 0.4 s ブロック中央値) | -9.02 | -20.23 | プロ全ブロック p5 | -14.64 | miss | 250/355 ブロック | 1-4k V/I 6.5 dB |
| eva-4 | buried/buried | 2:42.0 | A ドライのリード vs オケ+ウェット+ハモリ 損失(phon) | -10.18 | — | — | -15.09 | miss | 253/355 ブロック |  |
| eva-4 | buried/buried | 2:42.0 | I ESTOI（処理前→マスター、フレーズ） | 0.65 | 0.06 | 本家（分離ボーカル→原曲）p10 | 0.59 | miss | 6/11 フレーズ |  |
| eva-4 | buried/buried | 2:42.0 | I ASR 文字誤り率（処理前の書き起こし基準、マスター） | 0.84 | 1.50 | 本家（分離ボーカル→原曲）p90 | 0.84 | miss | 2/11 チャンク | 処理前「理性的であれヘイヘイヘイヘイボンボンヘイボンヘイ」/ マスター「理性的であれHey hey hey hey bo」 |
| six-g1 | sits_on_top/not_blended | 全体 | R_sits_loss: 部分ラウドネス損失 v7−本家（マスクされなさすぎ） | 1.22 | 1.00 | JND（v7 分離版−本家 分離） | — | HIT(abs) | — | v7 ステム−本家 0.33; ドライ再レンダー−本家 — |
| six-g1 | sits_on_top/not_blended | 全体 | R_sits_narrow: Side/Mid v7−本家 (dB) | -8.84 | -1.00 | JND（v7 分離版−本家 分離） | — | HIT(abs) | — | v7 ステム−本家 -7.76; ドライ再レンダー−本家 — |
| six-g1 | sits_on_top/not_blended | 全体 | R_sits_dry: フレーズ後の残り v7−本家 (dB) | 10.63 | -1.00 | JND（v7 分離版−本家 分離） | — | miss | — | v7 ステム−本家 1.96; ドライ再レンダー−本家 — |
| six-g2 | airy_fizzy/fizzy | 全体 | B fizz: Δ(8-16k − 1-4k) 処理後−処理前 (dB、参考) | 3.14 | — | （自MIX 基準は廃止） | — | — | — | Δflatness 0.019 |
| six-g2 | airy_fizzy/fizzy | 全体 | R_airy_abs: ボーカル自身 8-16k−1-4k v7−本家 (dB) | 3.71 | 1.00 | JND（v7 分離版−本家 分離） | — | HIT(abs) | — | v7 ステム−本家 3.21; ドライ再レンダー−本家 — |
| six-1 | harmony_prominent/too_strong | 0:29.0 | H ハモリ顕著度 p90 (phon, aux_double) | -78.54 | — | 参照分布なし | -54.76 | miss | 3 秒（この層） | リードからの音程 12.0 半音（pYIN、オクターブ誤りあり得る） |
| six-1 | harmony_prominent/too_strong | 0:29.0 | H ハモリ対リード音量 p90 (dB, aux_super_octave) | -7.44 | — | 参照分布なし | -7.61 | HIT(rel) | 20 秒（この層） | リードからの音程 11.9 半音（pYIN、オクターブ誤りあり得る） |
| six-2 | harmony_prominent/too_strong | 0:40.0 | H ハモリ顕著度 p90 (phon, choir) | -8.87 | — | 参照分布なし | 38.18 | miss | 46 秒（この層） | リードからの音程 -11.9 半音（pYIN、オクターブ誤りあり得る） |
| six-2 | harmony_prominent/too_strong | 0:40.0 | H ハモリ対リード音量 p90 (dB, choir) | 4.32 | — | 参照分布なし | 10.61 | miss | 42 秒（この層） | リードからの音程 -11.9 半音（pYIN、オクターブ誤りあり得る） |
| six-3 | step_change/too_far | 0:53.0 | D 段差 far_score(dB, 前後 3 s, リード+ウェット) | 10.09 | 1.00 | 音量 JND 1 dB | 5.85 | HIT(abs+rel) | 11 箇所 | Δlevel -4.0 / Δbright 0.8 / Δwet 6.8 @0:57.5 |
| six-4 | fx_dropout/missing_fx | 2:17.0 | G 演出終了→音符終了(s) | 1.50 | 0.10 | 時間積分 ~100 ms (Zwicker & Fastl) | — | HIT(abs) | 3/3 演出 | hard_autotune 2:17.3–2:20.2 → 音符終了 2:21.7 (+1.50 s); radio_voice 2:17.4–2:18.6 → 音符終了 2:19.5 (+0.89 s) |
| six-5 | delay_offbeat/off_grid | 2:24.0 | F ディレイ時間 − 最寄り音価(ms) | 21.40 | 6.00 | Friberg & Sundberg 1995: max(6 ms, 2.5% IOI) | — | HIT(abs) | 2/2 飛ばし | delay 235 ms (1/8T, 拍 640 ms); 実測エコー周期 81 ms → 誤差 -79 ms |
| six-6 | delay_offbeat/off_grid | 2:26.0 | F ディレイ時間 − 最寄り音価(ms) | 21.40 | 6.00 | Friberg & Sundberg 1995: max(6 ms, 2.5% IOI) | — | HIT(abs) | 2/2 飛ばし | delay 235 ms (1/8T, 拍 640 ms); 実測エコー周期 81 ms → 誤差 -79 ms |
| six-7 | consonant_attack/too_strong | 2:29.0 | C attack Δ子音/母音(ドライ, dB) | 4.68 | 7.66 | 比の絶対値 vs プロ p95 | 3.02 | HIT(rel) | 12/495 個 | @2:26.2 比の絶対値 -3.4 dB, V/I 8.4 dB, Δcrest 3.9 |
| six-8 | step_change/too_far | 2:43.0 | D 段差 far_score(dB, 前後 3 s, リード+ウェット) | 20.93 | 1.00 | 音量 JND 1 dB | 5.85 | HIT(abs+rel) | 2 箇所 | Δlevel -11.0 / Δbright 5.0 / Δwet 14.8 @2:42.0 |
| six-9 | sibilance/too_strong | 3:05.0 | C sibilant Δ子音/母音(ドライ, dB) | 3.81 | 1.69 | 比の絶対値 vs プロ p95 | 5.83 | miss | 40/92 個 | [Whisper で特定できず: 時刻窓で最悪の子音] @3:04.7 比の絶対値 -12.8 dB, V/I 2.0 dB, Δcrest 3.5 |
| lad-g1 | muddy/too_muddy | 全体 | R_muddy: bright_rel v7−本家 (dB) | -1.86 | -1.00 | JND（v7 分離版−本家 分離） | — | HIT(abs) | — | v7 ステム−本家 -1.29; ドライ再レンダー−本家 — |
| lad-1 | sibilance/crushed | 0:46.0 | C sibilant Δ子音/母音(ドライ, dB) | 4.58 | — | （量ではなく痕跡で判定: Q 行） | 2.06 | miss | 15/23 個 | [Whisper: 「さ」 0:46.6–0:46.8] @0:46.6 比の絶対値 -1.2 dB, V/I 11.0 dB, Δcrest 4.2 |
| lad-1 | sibilance/crushed | 0:46.0 | Q ディエッサー痕跡 raw 基準: (a)帯域減衰 dB | 4.88 | -3.00 | 1/3 oct の谷の可聴 ~3 dB (Bücklein 1981) | -0.15 | HIT(abs) | 17/23 個 | [Whisper: 「さ」 0:46.6–0:46.8] @0:46.6 ピーク 8699 Hz, (b)重心低下 3.6% (曲内 p90 6.3), (c)ゲインの谷 -3.4 dB (曲内 p10 -6.1), 残差 -7.0 dB, LSD 2.2 dB; うち下処理(prepared−raw) 帯域 0.6 / 谷 -3.4 dB |
| lad-1 | sibilance/crushed | 0:46.0 | Q ディエッサー痕跡 prepared 基準: (a)帯域減衰 dB | 4.31 | -3.00 | 1/3 oct の谷の可聴 ~3 dB (Bücklein 1981) | 0.54 | miss | 16/23 個 | [Whisper: 「さ」 0:46.6–0:46.8] @0:46.6 ピーク 8613 Hz, (b)重心低下 1.4% (曲内 p90 3.5), (c)ゲインの谷 -1.3 dB (曲内 p10 -2.0), 残差 -7.3 dB, LSD 1.9 dB |
| eva-g4 | step_change/timbre | 全体 | D 声質の段差: /Δbright/ ≥ JND の箇所 | 39.00 | 1.00 | 1 箇所以上（JND 1 dB） | — | HIT(abs) | 10.1 箇所/分 | うち v7 区間境界 ±1.5 s: 11; 時刻 0:22.8(+4.3), 0:26.2(-1.2), 0:29.5(+2.3), 0:34.2(-3.7), 0:40.0(+3.3), 0:43.0(-3.2), 0:46.0(+1.1), 0:50.0(+1.8); 参考: /Δlevel/≥1 dB のフレーム 348 |
| eva-g5 | sits_on_top/not_blended | 全体 | R_sits_loss: 部分ラウドネス損失 v7−本家（マスクされなさすぎ） | 4.13 | 1.00 | JND（v7 分離版−本家 分離） | — | HIT(abs) | — | v7 ステム−本家 2.94; ドライ再レンダー−本家 — |
| eva-g5 | sits_on_top/not_blended | 全体 | R_sits_narrow: Side/Mid v7−本家 (dB) | -6.49 | -1.00 | JND（v7 分離版−本家 分離） | — | HIT(abs) | — | v7 ステム−本家 -5.68; ドライ再レンダー−本家 — |
| eva-g5 | sits_on_top/not_blended | 全体 | R_sits_dry: フレーズ後の残り v7−本家 (dB) | -14.04 | -1.00 | JND（v7 分離版−本家 分離） | — | HIT(abs) | — | v7 ステム−本家 0.43; ドライ再レンダー−本家 — |
| eva-g6 | harmony_missing/missing | 全体 | H ハモリバス / リード (dB、曲全体) | -15.35 | — | 参照なし（本家のハモリは分離できない） | — | — | — | 現時点で検出不可（値のみ）。エバは録音ハモリ無し→Nectar 3 生成ハモリが 2mix に入る設計 |

## 指標ごとの hit 率

| 指標 | hit/件 |
|---|---|
| C attack Δ子音/母音 | 2/2 |
| R_space_tail | 0/1 |
| R_space_decay | 0/1 |
| R_buried | 0/1 |
| R_thin | 0/1 |
| C sibilant Δ子音/母音 | 1/3 |
| Q ディエッサー痕跡 raw 基準 | 2/2 |
| Q ディエッサー痕跡 prepared 基準 | 0/2 |
| A 部分ラウドネス損失 | 0/2 |
| A ドライのリード vs オケ+ウェット+ハモリ 損失 | 0/2 |
| I ESTOI（処理前→マスター、フレーズ） | 0/2 |
| I ASR 文字誤り率（処理前の書き起こし基準、マスター） | 0/2 |
| B bright_rel | 0/1 |
| R_sits_loss | 2/2 |
| R_sits_narrow | 2/2 |
| R_sits_dry | 1/2 |
| R_airy_abs | 1/1 |
| H ハモリ顕著度 p90 | 0/2 |
| H ハモリ対リード音量 p90 | 1/2 |
| D 段差 far_score | 2/2 |
| G 演出終了→音符終了 | 1/1 |
| F ディレイ時間 − 最寄り音価 | 2/2 |
| R_muddy | 1/1 |
| D 声質の段差 | 1/1 |

## 曲ごとの合否（本家との差の全体ルールのどれかが発火したら FAIL）

| 曲 | 判定 | 発火した全体ルール |
|---|---|---|
| Fall in Love | 判定不可 | 本家音源なし |
| エバ | FAIL | R_sits_loss(4.13), R_sits_narrow(-6.49), R_sits_dry(-14.04), R_muddy(-9.58) |
| シックスフィートアンダー | FAIL | R_sits_loss(1.22), R_sits_narrow(-8.84), R_muddy(-10.00), R_airy(2.60), R_airy_abs(3.71), R_space_tail(10.63) |
| ラブ&デストロイ | FAIL | R_sits_dry(-47.26), R_muddy(-1.86), R_airy(5.49), R_airy_abs(5.85) |

## 全体ルールの誤報（否定例）

全体ルールは本家（原曲を同じ BS-RoFormer で分離）との差を JND と比べる。v7 もマスターを同じモデルで分離した値を使う。同じ曲のプロカバー（ラブ&デストロイの梓川カバーのみ）で発火したら誤報。自MIX は凌の回答（2026-11-26「納得してない、天井にしないで」）により否定例・閾値の出典から外し、参考に値だけ載せる。Fall in Love は本家音源が無いので判定不可（fil-g2 も検証できない）。

| 全体ルール | v7（分離） | 同曲プロカバー（否定例） | 自MIX（参考・判定に使わない） |
|---|---|---|---|
| R_buried | Fall:— エバ:なし シックス:なし ラブ&デ:なし | ラブ&デ:発火 | シックス:発火 ラブ&デ:発火 |
| R_sits_loss | Fall:— エバ:発火 シックス:発火 ラブ&デ:なし | ラブ&デ:なし | シックス:なし ラブ&デ:なし |
| R_sits_narrow | Fall:— エバ:発火 シックス:発火 ラブ&デ:なし | ラブ&デ:なし | シックス:発火 ラブ&デ:なし |
| R_sits_dry | Fall:— エバ:発火 シックス:なし ラブ&デ:発火 | ラブ&デ:発火 | シックス:なし ラブ&デ:発火 |
| R_thin | Fall:— エバ:なし シックス:なし ラブ&デ:なし | ラブ&デ:なし | シックス:なし ラブ&デ:なし |
| R_muddy | Fall:— エバ:発火 シックス:発火 ラブ&デ:発火 | ラブ&デ:発火 | シックス:発火 ラブ&デ:発火 |
| R_airy | Fall:— エバ:なし シックス:発火 ラブ&デ:発火 | ラブ&デ:発火 | シックス:発火 ラブ&デ:発火 |
| R_airy_abs | Fall:— エバ:なし シックス:発火 ラブ&デ:発火 | ラブ&デ:発火 | シックス:発火 ラブ&デ:発火 |
| R_space_tail | Fall:— エバ:なし シックス:発火 ラブ&デ:なし | ラブ&デ:なし | シックス:発火 ラブ&デ:なし |
| R_space_decay | Fall:— エバ:なし シックス:なし ラブ&デ:なし | ラブ&デ:なし | シックス:発火 ラブ&デ:なし |

## 曲間の声色の並び（期待: Fall in Love が最も明るい／エアリー、シックスが次、ラブ&デストロイが最もこもる）

| 指標(大きい順) | 曲の並び |
|---|---|
| air_rel | Fall(-3.5) > エバ(-4.1) > シックス(-4.5) > ラブ&デ(-6.2) |
| hf_rel_abs | Fall(-12.0) > ラブ&デ(-12.2) > シックス(-12.6) > エバ(-14.4) |
| bright_rel | エバ(-0.2) > Fall(-2.0) > シックス(-3.6) > ラブ&デ(-7.5) |
| body_rel | ラブ&デ(2.9) > シックス(0.5) > Fall(-0.6) > エバ(-2.6) |

## Beat This! と v7 のテンポ

| 曲 | Beat This BPM | v7 BPM | v7 tempo_segments | 曲長(s) |
|---|---|---|---|---|
| Fall in Love | 125.0 | 122.4 | [[0.0, 122.4]] | 171.2 |
| エバ | 142.9 | 140.8 | [[0.0, 140.8]] | 231.5 |
| シックスフィートアンダー | 93.8 | 127.8 | [[0.0, 126.1], [444.5735, 191.4]] | 217.3 |
| ラブ&デストロイ | 142.9 | 140.6 | [[0.0, 140.6]] | 66.3 |
