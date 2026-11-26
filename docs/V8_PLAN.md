# v8 計画

v7（Gemini / Antigravity 版）から新エンジン v8 への作り直しの方針と順番です。
v7 の固定点: タグ `gemini-v7.0`（40a85cc）と `gemini-final`（8fa2575）。v7 の出力は `data/archive/gemini_v7_20261126/` に保存済み。

## 方針

- **正解**はユーザーの耳の評価（[docs/evaluations/](evaluations/README.md)）と Studio One での手直し履歴。**基準**は同じオケに乗った本家の声。プロのカバー音源は上限・下限の参考にとどめる。自 MIX は正解にも上限にも使わない（2026-11-26 凌「二つともそんなに納得してない、これを天井にはしないで」）。
- **目標はその曲のオケに対する相対値**で決める。公式オフボがある曲は「原曲 − オフボ」で声全体を、ない曲はカラオケ分離でリードとバッキングを分けて求める。
- **処理は 4 層**:
  1. 曲を通して固定のチェーン。実機 VST3 を CMA-ES で最適化し、目的関数は ST-ITO 的なもの、DiffVox の分布を事前分布に使う。
  2. フレーズ単位の部分ラウドネスに基づく音量ライド。区間ごとの EQ・サチュレーション切替は廃止し、Studio One のクリップゲイン／オートメーションとして書き出す。
  3. 音節単位で子音・アタックを双方向に（潰れていれば持ち上げ、強すぎれば抑える）整える。
  4. 空間はオケ自身の残響から決め、ディレイは Beat This! の拍格子に合わせる。
- **根拠のない定数を書かない。** 値には出典（ユーザー評価・計測・論文・既存プロファイル）を残す。

## 順番（各段階 = 別ブランチ / PR）

| # | 段階 | 合格の目安 |
|---|---|---|
| 1 | 戻せる状態（固定タグ・v7 出力の保存・Gemini 版 worktree・`--engine v7\|v8`・来歴記録・開発基盤） | v7 のエバが worktree で再現できる |
| 2 | 正解データと評価 | 指摘（issues.yaml）を当てられる指標だけ採用。v7 のエバが不合格になること |
| 3 | 子音が潰れる／強すぎる処理段の特定 | エバ 0:20、ラブ&デストロイ 0:46、シックス 2:29・3:05、Fall in Love 2:34 の原因段が特定できる |
| 4 | 拍・小節・構造・音節の解析作り直し | |
| 5 | オケ相対の正解 | |
| 6 | 固定チェーンと芯 | |
| 7 | フレーズ音量ライドと Studio One 書き出し | |
| 8 | 音節単位の子音 | |
| 9 | 空間と演出 | |
| 10 | ハモリと提案トラック | |
| 11 | v7 撤去（ユーザー確認後） | |

- v7 と v8 の 4 曲 A/B をユーザーが試聴して OK したものだけ main へ入れる。
- 段階ごとにセッションを分ける（運用は [CLAUDE.md](../CLAUDE.md)）。

## Phase 1 で用意したもの

| もの | 場所 |
|---|---|
| エンジン切替 | `mix-auto mix … --engine v7\|v8`（既定 v7。v8 は「未実装」で停止） |
| 来歴 | `mix_report.json` の `engine` / `git_commit` / `git_dirty` / `profiles_json` / `profiles_sha256` |
| v8 骨格 | `src/mix_automation/v8/` |
| Gemini 版の実行環境 | `C:\MIX_automation_gemini`（worktree @ `gemini-final`、専用 `.venv`、出力は `C:\MIX_automation_gemini\outputs`） |
| 検証 | `scripts/dev_check.py` |
| 解析 | `scripts/analysis/`（区間スペクトル差分・時刻→区間・区間バランス） |
| 正解データ | `docs/evaluations/`、下書き生成 `scripts/evaluations/parse_notes.py` |
| サブエージェント | `.claude/agents/implementer.md`・`worker.md` |

## Phase 2 で用意したもの（採否はレビューで決める）

| もの | 場所 |
|---|---|
| 指標（純関数・出典コメント付き） | `src/mix_automation/v8/metrics/`（loudness=A 部分ラウドネス, spectral=B 声色, consonants=C 子音, steps=D 段差, space=E 空間, timing=F 拍ずれ・G 演出抜け, harmony=H ハモリ） |
| 合成信号テスト | `tests/test_v8_metrics.py`（各指標の向き） |
| 参照分布 | `docs/evaluations/reference_stats.json`（`scripts/analysis/build_reference_stats.py`） |
| 判定表 | `docs/evaluations/metric_validation.md`（`scripts/analysis/evaluate_issues.py --write-md`） |

検証で分かったこと（2026-11-26、v7 の 4 曲）:

- 当たった: 子音アタックの強調（fil-1, six-7。処理後−処理前の「子音／母音」比）、段差（six-3, six-8）、演出抜け（six-4）、ディレイの拍ずれ（six-5/6）。いずれも局所・処理前との差で見る指標。
- 当たらない（現時点で検出不可）: 全体の埋もれ・軽さ・こもり・エアリー・ジリつき・空間過多・載ってる、局所の埋もれ（eva-2/4）、歯擦音（潰れ・強すぎとも）。全体指標はプロ分布（分離ステム）が広すぎ、自 MIX でも誤報が出る。
- v7 の書き出しの癖: シックスの `VocalBus.wav` は −5.3 dB ピーク正規化されている（マスターから推定して補正）。`02_mainvocal_prepared.wav` は演出区間が抜かれている（演出トラックを足し戻して処理前とする）。
- v7 のテンポ: シックスは Beat This! 93.8 BPM に対し v7 126.1（2 番目のセグメントは 444.6 s 開始で曲長 217 s を超える）。他 3 曲は 2〜3 BPM 差。
