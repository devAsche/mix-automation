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

## Phase 2 で用意したもの

| もの | 場所 |
|---|---|
| 指標（純関数・出典コメント付き） | `src/mix_automation/v8/metrics/`（loudness=A 部分ラウドネス, spectral=B 声色, consonants=C 子音・Q ディエッサー痕跡, steps=D 段差, space=E 空間, timing=F 拍ずれ・G 演出抜け, harmony=H ハモリ, intelligibility=I 歌詞明瞭度） |
| 合成信号テスト | `tests/test_v8_metrics.py`（各指標の向き、部分ラウドネスのマスク閾値） |
| 参照分布 | `docs/evaluations/reference_stats.json`（`scripts/analysis/build_reference_stats.py`） |
| 判定表 | `docs/evaluations/metric_validation.md`（`scripts/analysis/evaluate_issues.py --write-md`。手で編集しない） |
| v7 の中間ステム | `mix-auto mix … --export-debug-stems`（リードのドライ・ウェット・ハモリバス・aux。音は変わらない。4 曲でアーカイブとヌル −61〜−66 dB） |

## Phase 2 の結論（2026-11-26 レビュー）

v7 のエバは FAIL になる。理由は Side/Mid の狭さと「マスクされなさすぎ」（本家比）で、凌さんの「声が浮いていた」（eva-g5）と一致する。局所では「し」の潰れ（eva-1）と、パートごとの声質の段差（eva-g4）も当たる。一方、エバの「埋もれ」「軽い」「空間過多」はまだ当てられない。合格の目安は一部だけ達成とし、残りは同じ曲の人間のプロカバーが揃う Phase 5 で詰める。

| 区分 | 指標 | 当てた指摘 | 備考 |
|---|---|---|---|
| 採用 | D 段差 far_score／声質の段差（処理後−処理前） | six-3, six-8, eva-g4 | 歌唱の変化を除くため処理前との差で見る |
| 採用 | G 演出終了→音符終了 | six-4 | 基準 0.1 s（Zwicker & Fastl の時間積分） |
| 採用 | F ディレイ時間 − 拍格子の音価（Beat This!） | six-5, six-6 | 許容 max(6 ms, 2.5% IOI)（Friberg & Sundberg 1995） |
| 採用 | Q ディエッサー痕跡（raw 基準: 帯域減衰・重心低下・ゲインの谷） | eva-1, lad-1（弱） | 「潰れ」= ディエッサー過多（凌、2026-11-26）。eva-1 は下処理段（prepared−raw）で −9.9 dB |
| 採用（曲内基準のみ） | C 子音アタックの Δ子音／母音 | fil-1, six-7 | 絶対閾値は未確定。fil-1 のアドリブはハモリ側のステムにある |
| 暫定 | R_sits_narrow／R_sits_loss（本家との差） | eva-g5, six-g1 | 幅は歌い手に依らない比較なので本家がボカロでも使える見込み（推測） |
| 暫定 | H ハモリ対リード音量（＋音程） | six-1 | six-2 は外れ。参照分布なし |
| 不採用（検出不可） | 埋もれ（部分ラウドネス・ESTOI・ASR 文字誤り率）、軽さ、空間の量、こもり・エアリー（本家比）、歯擦音の量 | — | 本家のエバ・シックスはボカロ（flower・GUMI）で声色の比較が成り立たない。同曲プロカバー（梓川のラブ&デストロイ）で誤報。エバの空間は本家より 14 dB 乾いており、「過剰」は量ではなく種類の不一致の可能性（推測） |

Phase 5 に必要な素材: Fall in Love の本家音源、4 曲それぞれの人間のプロカバー（できれば複数。カバー間のばらつきを閾値にするため）。

- 自 MIX は判定・否定例・閾値の出典から外した（表には参考値のみ）。
- v7 の書き出しの癖: シックスとラブ&デストロイの `VocalBus.wav` はピーク正規化されている（シックス −5.3 dB、ラブ&デストロイ 0.89）。`02_mainvocal_prepared.wav` は演出区間が抜かれている（演出トラックを足し戻して処理前とする）。
- Whisper の単語タイムスタンプは、エバ 0:20 では幻覚して使えなかった。歯擦音の検出がある場所だけ位置に採用し、無い場合は時刻窓に戻す。
- v7 のテンポ: シックスは Beat This! 93.8 BPM に対し v7 126.1（2 番目のセグメントは 444.6 s 開始で曲長 217 s を超える）。他 3 曲は 2〜3 BPM 差。
