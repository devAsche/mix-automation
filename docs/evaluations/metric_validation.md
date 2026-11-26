# 指標の検証（Phase 2）

`scripts/analysis/evaluate_issues.py --write-md` の出力。指標の定義と出典は `src/mix_automation/v8/metrics/`、参照分布は `reference_stats.json`（`scripts/analysis/build_reference_stats.py`）。
参照: プロ 87 曲（BS-RoFormer 分離済みステム）、自MIX 2 件。

## 指摘 × 指標

判定: HIT(abs) = 絶対基準（プロ分布・自MIX・JND）を超えた、HIT(rel) = 同じ曲の残りの時間の 10/90 パーセンタイルを超えた。時刻窓は ±1 s（approx は ±3 s）、「～／から」は終わりを +3 s。閾値は issues.yaml からは決めていない。「同程度」= 同じ曲で同じくらい悪い箇所の数（指摘されていない箇所は誤報に数えない）。

| id | 分類/向き | 時刻 | 指標 | 値 | 絶対閾値 | 根拠 | 曲内閾値 | 判定 | 同程度 | 補足 |
|---|---|---|---|---|---|---|---|---|---|---|
| fil-g1 | airy_fizzy/too_airy | 全体 | G_airy: (V-I)8-16k − (V-I)1-4k | -3.46 | 6.27 | プロ p95 | — | miss | — |  |
| fil-g1 | airy_fizzy/too_airy | 全体 | G_airy_abs: ボーカル自身 8-16k − 1-4k | -11.98 | -6.69 | プロ p95 | — | miss | — |  |
| fil-1 | consonant_attack/too_strong | 2:34.0 | C attack Δ子音/母音(dB) | 13.53 | 6.72 | 自MIX Δ p95 | 4.21 | HIT(abs+rel) | 1/422 個 | @2:33.3 V/I -9.7 dB (プロ p5 -13.2), Δcrest 11.1 |
| fil-1 | consonant_attack/too_strong | 2:34.0 | C attack Δ子音/母音 raw基準(dB) | 13.68 | — | （自MIX の raw 基準 Δ は未計算） | 4.72 | HIT(rel) | 1/422 個 | @2:33.3 うち下処理(prepared−raw) 0.2 dB |
| eva-g1 | space_excess/too_much | 全体 | G_space_tail: フレーズ後 0.5 s の残り（dB） | -39.26 | -13.28 | プロ p95 | — | miss | — |  |
| eva-g1 | space_excess/too_much | 全体 | G_space_decay: フレーズ終わりの減衰時間（T20→60 dB） | 0.57 | 2.22 | プロ p95 | — | miss | — |  |
| eva-g1 | space_excess/too_much | 全体 | G_space_vs_inst: 減衰時間 ÷ オケのブラインド RT60 | 0.12 | 0.69 | プロ p95 | — | miss | — |  |
| eva-g2 | buried/buried | 全体 | G_buried: 部分ラウドネス損失（0.4 s ブロック中央値） | -10.09 | -13.15 | プロ p5 | — | miss | — |  |
| eva-g3 | thin_light/too_light | 全体 | G_thin: (V-I)250-500 − (V-I)1-4k | -2.63 | -6.93 | プロ p5 | — | miss | — |  |
| eva-1 | sibilance/crushed | 0:20.0 | C sibilant Δ子音/母音(dB) | 3.19 | 0.46 | 自MIX Δ p5 | 0.67 | miss | 31/52 個 | @0:21.7 V/I 8.4 dB (プロ p5 -0.7), Δcrest 3.3 |
| eva-1 | sibilance/crushed | 0:20.0 | C sibilant Δ子音/母音 raw基準(dB) | 4.26 | — | （自MIX の raw 基準 Δ は未計算） | -0.34 | miss | 32/52 個 | @0:21.7 うち下処理(prepared−raw) 1.1 dB |
| eva-2 | buried/buried | 1:31.0 | A 部分ラウドネス損失(phon, 窓内 0.4 s ブロック中央値) | -10.17 | -20.23 | プロ全ブロック p5 | -14.64 | miss | 172/355 ブロック | 1-4k V/I 1.9 dB |
| eva-3 | muddy/too_muddy | 2:01.0 | B bright_rel (6 s 窓) | 0.60 | -1.00 | 以降−以前 の差 ≤ −1 dB(JND) | -3.75 | miss | 116/164 窓 | 以降−以前 -0.98 dB |
| eva-4 | buried/buried | 2:42.0 | A 部分ラウドネス損失(phon, 窓内 0.4 s ブロック中央値) | -9.02 | -20.23 | プロ全ブロック p5 | -14.64 | miss | 250/355 ブロック | 1-4k V/I 6.5 dB |
| six-g1 | sits_on_top/not_blended | 全体 | G_sits_loss: 同上が小さすぎる（マスクされなさすぎ） | -12.89 | -7.13 | プロ p95 | — | miss | — |  |
| six-g1 | sits_on_top/not_blended | 全体 | G_sits_narrow: ボーカル Side/Mid（狭さ） | -10.88 | -16.81 | プロ p5 | — | miss | — |  |
| six-g2 | airy_fizzy/fizzy | 全体 | B fizz: Δ(8-16k − 1-4k) 処理後−処理前 (dB) | 3.14 | 6.59 | 自MIX の同じ差の最大 | — | miss | — | Δflatness 0.019 |
| six-g2 | airy_fizzy/fizzy | 全体 | G_airy_abs: ボーカル自身 8-16k − 1-4k | -12.58 | -6.69 | プロ p95 | — | miss | — |  |
| six-1 | harmony_prominent/too_strong | 0:29.0 | H ハモリ顕著度 p90 (phon, aux_double) | -78.55 | — | 参照分布なし | -54.76 | miss | 3 秒（この層） | リードからの音程 12.0 半音（pYIN、オクターブ誤りあり得る） |
| six-1 | harmony_prominent/too_strong | 0:29.0 | H ハモリ対リード音量 p90 (dB, aux_super_octave) | -7.44 | — | 参照分布なし | -7.61 | HIT(rel) | 20 秒（この層） | リードからの音程 11.9 半音（pYIN、オクターブ誤りあり得る） |
| six-2 | harmony_prominent/too_strong | 0:40.0 | H ハモリ顕著度 p90 (phon, choir) | -8.87 | — | 参照分布なし | 38.17 | miss | 46 秒（この層） | リードからの音程 -11.9 半音（pYIN、オクターブ誤りあり得る） |
| six-2 | harmony_prominent/too_strong | 0:40.0 | H ハモリ対リード音量 p90 (dB, choir) | 4.32 | — | 参照分布なし | 10.61 | miss | 42 秒（この層） | リードからの音程 -11.9 半音（pYIN、オクターブ誤りあり得る） |
| six-3 | step_change/too_far | 0:53.0 | D 段差 far_score(dB, 前後 3 s, リード推定) | 11.72 | 1.00 | 音量 JND 1 dB | 5.10 | HIT(abs+rel) | 8 箇所 | Δlevel -4.1 / Δbright 0.9 / Δwet 8.6 @0:57.5 |
| six-4 | fx_dropout/missing_fx | 2:17.0 | G 演出終了→音符終了(s) | 1.50 | 0.10 | 時間積分 ~100 ms (Zwicker & Fastl) | — | HIT(abs) | 3/3 演出 | hard_autotune 2:17.3–2:20.2 → 音符終了 2:21.7 (+1.50 s); radio_voice 2:17.4–2:18.6 → 音符終了 2:19.5 (+0.89 s) |
| six-5 | delay_offbeat/off_grid | 2:24.0 | F ディレイ時間 − 最寄り音価(ms) | 21.40 | 6.00 | Friberg & Sundberg 1995: max(6 ms, 2.5% IOI) | — | HIT(abs) | 2/2 飛ばし | delay 235 ms (1/8T, 拍 640 ms); 実測エコー周期 81 ms → 誤差 -79 ms |
| six-6 | delay_offbeat/off_grid | 2:26.0 | F ディレイ時間 − 最寄り音価(ms) | 21.40 | 6.00 | Friberg & Sundberg 1995: max(6 ms, 2.5% IOI) | — | HIT(abs) | 2/2 飛ばし | delay 235 ms (1/8T, 拍 640 ms); 実測エコー周期 81 ms → 誤差 -79 ms |
| six-7 | consonant_attack/too_strong | 2:29.0 | C attack Δ子音/母音(dB) | 4.83 | 6.72 | 自MIX Δ p95 | 3.28 | HIT(rel) | 22/495 個 | @2:26.2 V/I 9.3 dB (プロ p5 -13.2), Δcrest 4.2 |
| six-7 | consonant_attack/too_strong | 2:29.0 | C attack Δ子音/母音 raw基準(dB) | 5.29 | — | （自MIX の raw 基準 Δ は未計算） | 3.40 | HIT(rel) | 19/495 個 | @2:27.7 うち下処理(prepared−raw) 5.7 dB |
| six-8 | step_change/too_far | 2:43.0 | D 段差 far_score(dB, 前後 3 s, リード推定) | 20.48 | 1.00 | 音量 JND 1 dB | 5.10 | HIT(abs+rel) | 2 箇所 | Δlevel -10.7 / Δbright 4.7 / Δwet 14.5 @2:42.0 |
| six-9 | sibilance/too_strong | 3:05.0 | C sibilant Δ子音/母音(dB) | 7.52 | 12.98 | 自MIX Δ p95 | 8.11 | miss | 13/83 個 | @3:04.6 V/I 3.4 dB (プロ p5 -0.7), Δcrest 4.7 |
| six-9 | sibilance/too_strong | 3:05.0 | C sibilant Δ子音/母音 raw基準(dB) | 6.65 | — | （自MIX の raw 基準 Δ は未計算） | 8.95 | miss | 22/83 個 | @3:04.6 うち下処理(prepared−raw) -0.9 dB |
| lad-g1 | muddy/too_muddy | 全体 | G_muddy: (V-I)4-16k − (V-I)250-500 | -7.45 | -9.49 | プロ p5 | — | miss | — |  |
| lad-1 | sibilance/crushed | 0:46.0 | C sibilant Δ子音/母音(dB) | 3.88 | 0.46 | 自MIX Δ p5 | 1.48 | miss | 10/21 個 | @0:46.6 V/I 12.1 dB (プロ p5 -0.7), Δcrest 3.9 |
| lad-1 | sibilance/crushed | 0:46.0 | C sibilant Δ子音/母音 raw基準(dB) | 3.30 | — | （自MIX の raw 基準 Δ は未計算） | 0.36 | miss | 10/21 個 | @0:45.1 うち下処理(prepared−raw) -1.8 dB |

## 指標ごとの hit 率

| 指標 | hit/件 |
|---|---|
| G_airy | 0/1 |
| G_airy_abs | 0/2 |
| C attack Δ子音/母音 | 2/2 |
| C attack Δ子音/母音 raw基準 | 2/2 |
| G_space_tail | 0/1 |
| G_space_decay | 0/1 |
| G_space_vs_inst | 0/1 |
| G_buried | 0/1 |
| G_thin | 0/1 |
| C sibilant Δ子音/母音 | 0/3 |
| C sibilant Δ子音/母音 raw基準 | 0/3 |
| A 部分ラウドネス損失 | 0/2 |
| B bright_rel | 0/1 |
| G_sits_loss | 0/1 |
| G_sits_narrow | 0/1 |
| B fizz | 0/1 |
| H ハモリ顕著度 p90 | 0/2 |
| H ハモリ対リード音量 p90 | 1/2 |
| D 段差 far_score | 2/2 |
| G 演出終了→音符終了 | 1/1 |
| F ディレイ時間 − 最寄り音価 | 2/2 |
| G_muddy | 0/1 |

## 曲ごとの合否（全体ルールのどれかが発火したら FAIL）

| 曲 | 判定 | 発火した全体ルール |
|---|---|---|
| Fall in Love | PASS | — |
| エバ | FAIL | G_sits_narrow(-18.71) |
| シックスフィートアンダー | PASS | — |
| ラブ&デストロイ | PASS | — |

## 全体ルールの誤報（否定例）

プロ LOO = その曲を除いた分布の p5/p95 で判定（構造上 約 10%）。自MIX・FiL（fil-g2: 距離感 OK）で発火したら誤報。v7 分離版 = v7 マスターを同じ BS-RoFormer で分離して計算（真のステムとの差 = 分離バイアス）。

| 全体ルール | プロ LOO 発火 | 自MIX | FiL(距離OK) | v7 分離版 |
|---|---|---|---|---|
| G_buried | 5/87 | シックス:発火 ラブ&デ:なし | OK | Fal:なし エバ:なし シック:なし ラブ&:なし |
| G_sits_loss | 5/87 | シックス:なし ラブ&デ:なし | OK | Fal:なし エバ:なし シック:なし ラブ&:なし |
| G_sits_narrow | 6/87 | シックス:なし ラブ&デ:なし | OK | Fal:なし エバ:発火 シック:なし ラブ&:なし |
| G_thin | 5/87 | シックス:なし ラブ&デ:なし | — | Fal:なし エバ:なし シック:なし ラブ&:なし |
| G_muddy | 5/87 | シックス:なし ラブ&デ:発火 | — | Fal:なし エバ:なし シック:なし ラブ&:なし |
| G_airy | 6/87 | シックス:なし ラブ&デ:なし | — | Fal:なし エバ:なし シック:なし ラブ&:なし |
| G_airy_abs | 5/87 | シックス:なし ラブ&デ:なし | — | Fal:なし エバ:なし シック:なし ラブ&:なし |
| G_space_tail | 5/87 | シックス:なし ラブ&デ:なし | OK | Fal:なし エバ:なし シック:なし ラブ&:なし |
| G_space_decay | 5/87 | シックス:なし ラブ&デ:なし | OK | Fal:なし エバ:なし シック:なし ラブ&:なし |
| G_space_vs_inst | 1/20 | シックス:なし ラブ&デ:なし | OK | Fal:なし エバ:なし シック:なし ラブ&:なし |

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
