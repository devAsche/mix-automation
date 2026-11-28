# R5b 1': サビの埋もれを処理で直す — 実装仕様（Claude 作業、2026-11-28）

方針: 区間ごとの音量（フェーダー）ではなく、コンプの入り方・時間特性・サビだけの処理で直す。フェーダーは最後の仕上げとして ±1.5 dB 以内、罰つき。
根拠: rulebook 2.0 末行 -c0bHeO5ISs@10:17（区間でトラックを分けてコンプのかかり方を揃える）、2.4、2.7（1176 系パラレル約 8%、子音のアタック）、2.9 末行 JUXJShxfzKQ@43:06（サビはフェーダーの代わりにダブル）、2.13、5.2 ZKot7JyrTZA@10:37（サビ頭の数語だけ大きく）、1 の 4-GZf4N1cso@4:10。

R5（既存）の動作は変えない。新しい動作は「r5b モード」= 区間地図（SectionMap）が渡されたときだけ。r5b でないときの出力は今とビット一致（テスト）。

## 1. 区間地図 `v8/sections.py`

```
@dataclass
class SectionMap:
    sections: list[dict]   # {label, type, t0, t1, active_s, level_db, trim_db, restore_db}
    type_idx: np.ndarray   # int8 per sample: 0 verse, 1 chorus, 2 quiet, -1 none
    chorus_env: np.ndarray # float32 0..1, chorus = 1, raised-cosine ramps of 1 beat centred on boundaries
    trim_db: np.ndarray    # float32 per sample, same ramps
    restore_db: np.ndarray # float32 per sample, same ramps
    head_env: np.ndarray   # float32 0..1: chorus-head windows (below)
    def cut(self, lay) -> "SectionMap"   # every per-sample array through lay.cut(arr, lay.orig_starts())
```

`build_section_map(lead_front, inst, sr, tempo_bpm)`:
- 区間: `dsp.vst3_optimizer.detect_song_sections(lead_mono, inst_mono, sr)`。型: Chorus→chorus、Verse_PreChorus→verse、Quiet_Bridge_Verse→quiet、Instrumental→none（A・B メロは分けない）。
- **境目の吸着**: 凌の発言と照らすと、構造解析の境目は 1〜2 s ずれる（SFU のサビ入り 0:53 → 検出 0:54.3、エバ 1:31 の音数の変化 → 1:29.7、エバの歌い出し 0:20 → 0:22.3）。そこで各境目を、±3 s 以内で最も近い「リードの無音（P.56 不活動が 0.25 s 以上続く所）」に吸着させ、無音の終わり（次のフレーズの頭）の 0.1 s 前に置く。無音が無ければ動かさない。ランプは無音の中に入るので聞こえにくい。吸着前後の境目を mix_report に残す。
- level_db = 区間内のリードの P.56 アクティブレベル（`dsp` / `metrics._core` の既存関数）。active_s < 2 s の区間は trim 0。
- L_ref = active_s で重みを付けた level_db の平均。trim_db = clip(L_ref − level_db, −6, +6)。
- restore_db = clip(−trim_db, −1.5, +1.5)（コンプ後に区間差を最大 ±1.5 dB まで戻す）。
- head_env: chorus 区間ごとに、区間開始後にリードが最初にアクティブになった時刻 t_on から 1.5 s を 1、立ち上がり 30 ms・立ち下がり 300 ms（区間の前半に限る）。
- 境目のつなぎ: 1 拍 = 60 / tempo_bpm 秒の raised-cosine（境目を中心に）。

## 2. 新しいつまみ（`knobs.py` の KNOB_SPECS に追加、既定値は R5 相当）

| つまみ | 範囲 | 既定 | 段 | 中身 |
|---|---|---|---|---|
| vca_thr_rel_db | −24 … 0 | −10 | A | VCA の compression_db = VCA 入力の P.56 アクティブレベル + この値 |
| vca_ratio | 1.5 … 8 | 4 | A | VCA ratio（rulebook 2:1〜4:1、深めまで。実機は離散値 '1.5:1' 等、最も近い値に丸める） |
| vca_attack | 0.25 … 4.0 | 0.4 | A | VCA attack（実機の値。0.25〜4.0、刻み 0.01。単位は表示に無い＝倍率とみる） |
| vca_release | 0.25 … 4.0 | 0.85 | A | VCA release（同上） |
| vca_makeup_db | 0 … +12 | 0 | A | VCA output_db |
| la2a_peak_reduct | 0 … 100 | 50 | A | LA-2A Peak Reduction（スレッショルドに当たる。入力は前段で声の音量に揃っているので、声の音量からの相対になる） |
| la2a_gain | 0 … 100 | 35 | A | LA-2A Gain（メイクアップ） |
| par_comp_chorus_extra | 0 … 0.2 | 0 | A | サビの並列コンプ混ぜ量 = par_comp_mix + この値 |
| doubler_chorus_extra_db | 0 … 6 | 0 | B | サビのダブラーセンド = doubler_send_db + この値 |
| chorus_head_db | 0 … 2 | 0 | B | サビ頭（head_env）のゲイン |
| fader_chorus_db | −1.5 … 1.5 | 0 | B | サビのフェーダー（verse 基準） |
| fader_quiet_db | −1.5 … 1.5 | 0 | B | quiet のフェーダー |

- vca_attack / vca_release: 実機（Softube VCA Compressor）の parameter は attack・release とも 0.25〜4.0（刻み 0.01、表示に単位なし）。ms への換算は、ステップ入力でアタック・リリースの時定数を測る小さなスクリプト `scripts/analysis/vca_time_constants.py` で 0.25 / 0.4 / 1 / 2 / 4 の 5 点を出し、README に表で載せる（rulebook の目安 Attack 15〜30 ms・Release 100〜170 ms が範囲内かを確かめる）。探索は log 尺度（to_unit で log）にする。
- 初期値: R5 最終（data/outputs_v8r5/knobs/*.json と各 render の mix_report の lead_comp1_vca / lead_comp2_la2a の calibration 値）から換算した値を `scripts/match/r5b_init.py` で作る。較正は初期値づくりにだけ使い、r5b の探索中はしない（README に書く）。

## 3. チェーンの変更（`chain.py`、r5b モードのときだけ）

lead_tone(x, mask, sr, knobs, log, smap=None):
1. 引き算 EQ、pre_level は今のまま。
2. **区間トリム**: x *= 10^(smap.trim_db/20)（VCA の前）。
3. VCA: configure_softube_vca(threshold = active_level(x) + vca_thr_rel_db, ratio, attack, release, output_db = vca_makeup_db)。較正しない。GR は今の測り方（_static_gain_comp_gr）で測って log に残す。
4. LA-2A: peak_reduct = la2a_peak_reduct、gain = la2a_gain。較正しない。GR を log。
5. 並列コンプ: mix_env = par_comp_mix + par_comp_chorus_extra × chorus_env。`dsp.parallel_compress` に mix_env（per-sample）を渡せるようにする（スカラーのときは今と同じ）。
6. **戻し**: x *= 10^(smap.restore_db/20)。その後の lead_post_comp_trim（全体の音量合わせ）は今のまま。
7. Pultec 以降は今のまま（Pultec は R5 固定）。

lead_ride の後（新関数 `section_post(x, smap, knobs, log)`）: x *= 10^((chorus_head_db × head_env + fader[type_idx 由来, ramped])/20)。fader は verse 0・chorus fader_chorus_db・quiet fader_quiet_db を chorus_env と同じ形のランプでつなぐ（type ごとの env を作る）。
build_sends: doubler の wet に 10^(doubler_chorus_extra_db × chorus_env / 20)。

engine: front の出力と inst から SectionMap を作り（tempo は既存の推定）、lead_tone / section_post / build_sends に渡す。オプション `r5b: bool`（CLI `--r5b`）。mix_report に sections 表（label, type, t0, t1, level_db, trim_db, restore_db）と GR を残す。

## 4. 目的関数（`match.py`、`SongMatch(r5b=True)`）

- キャッシュ: data/processed/v8r_sections/<曲>/（*_r5b.json）。R5 のキャッシュは読むだけ。
- **区間の抜粋を足す**: 型ごと（verse・chorus・quiet）に、その型の区間からメインだけのフレーム（リード活動 & 非リード不活動）が最も多い 6 s の窓を 1 つずつ選び、既存の Region と同じ形（前後パディング付き）で Layout に足す（既存の領域と重なるなら足さない）。型の区間が無い曲はその型を飛ばす。
- SectionMap は全曲の front 出力で作り、`smap.cut(lay)` で抜粋の時間に合わせる。
- 特徴 "sec": 型ごとの kVI（K 重み、メインだけのフレーム、section_targets.py と同じ定義）を求め、[chorus − verse, quiet − verse] のベクトル。プロも同じ原時刻を orig_to_cover で写した窓で測る。term_raw は既存のベクトル項と同じ（平均二乗差）。スケールは fallback（SD 1.5 dB）で固定。重み 2（lu と同じ）。
- 罰 "fader": (fader_chorus_db/1.5)² + (fader_quiet_db/1.5)² を重み 0.5 で距離に足す（プロ側の値は無いので distance の外で加算し、評価記録に別に残す）。
- STAGE_A_R5B = (vca_thr_rel_db, vca_ratio, vca_attack, vca_release, vca_makeup_db, la2a_peak_reduct, la2a_gain, pre_level_strength, par_comp_mix, par_comp_chorus_extra)
- STAGE_B_R5B = STAGE_B + (doubler_chorus_extra_db, chorus_head_db, fader_chorus_db, fader_quiet_db)
- `scripts/match/run_match_fast.py --r5b`（R5 と同じ流れ・予算指定）。1 評価の時間と、1 曲あたりの探索時間を記録に残す。

## 5. テスト

- r5b でないときビット一致（SectionMap=None）。
- build_section_map: trim の重み付き平均 0、clip、restore の ±1.5、ランプ長、head_env の位置。
- parallel_compress: スカラーと定数 env が一致。
- section_post: fader 0・head 0 で恒等。
- match: 区間抜粋の選び方（合成データ）、"sec" 特徴の向き。
