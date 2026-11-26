# ユーザー評価（正解データ）

v8 の「正解」は、ユーザー（凌）の耳による評価・自 MIX・Studio One での手直し履歴を主とします（[V8_PLAN.md](../V8_PLAN.md)）。
このフォルダは、そのうち耳による評価を保存・構造化する場所です。

| ファイル | 内容 |
|---|---|
| `YYYY-MM-DD_曲の評価.txt` | ユーザーのメモの原文（一字一句そのまま。編集しない） |
| `issues.yaml` | 原文を 1 指摘 = 1 項目に構造化したもの。評価指標の検証（指摘を当てられるか）に使う |
| `metric_validation.md` | Phase 2: 指摘 × 指標の判定表・hit 率・誤報・v7 の合否（`scripts/analysis/evaluate_issues.py --write-md` の出力） |
| `reference_stats.json` | Phase 2: 指標の参照分布（プロ 87 曲の分離ステム・自 MIX 2 曲・v7 分離版、`scripts/analysis/build_reference_stats.py`） |

新しいメモが来たら:

1. 原文を `docs/evaluations/YYYY-MM-DD_曲の評価.txt` にそのままコピーする
2. `.venv\Scripts\python.exe scripts\evaluations\parse_notes.py <原文> -o docs\evaluations\draft.yaml` で下書きを作る
3. category / direction を人が確認・修正し（下書きの `draft: true` はキーワード推定）、`issues.yaml` に追記する
4. `tests/test_evaluations.py` が通ること（quote が原文に含まれるか、category が定義済みかを検査）

時刻と区間の対応は `scripts/analysis/time_to_section.py <曲> --issues docs/evaluations/issues.yaml` で確認できます。

## 項目

| 項目 | 必須 | 説明 |
|---|---|---|
| `id` | ○ | 曲の略号 + 連番。全体指摘は `-gN`（例: `eva-1`, `eva-g2`） |
| `song` | ○ | `mix_report.json` の `song_name` と同じ表記（例: `ラブ&デストロイ`） |
| `time_sec` | △ | 指摘の開始時刻（秒）。ユーザーの「m:ss～」は**その時刻から**の意味 |
| `start` / `end` | △ | 範囲がはっきりしている場合（秒） |
| `scope: global` | △ | 曲全体への指摘。`time_sec` と排他 |
| `approx` |  | 「くらい」「あたり」「直前」など時刻があいまいな場合 `true` |
| `lyric` |  | 該当歌詞。特定の音（「し」など）は括弧で添える |
| `category` | ○ | 下の分類 |
| `direction` | ○ | どちらに外れているか（下の表） |
| `related` |  | 同じ一文に含まれる副次的な分類 |
| `polarity` |  | 良い評価（守るべき点）は `positive`。既定は問題指摘 |
| `note` |  | 補足（解釈、同じ指摘の重複など） |
| `quote` | ○ | ユーザーの言葉そのまま。複数行をまとめたときは ` / ` で区切る |

## 分類（category）

| category | 定義 | 主な direction |
|---|---|---|
| `sibilance` | サ行・シ・チなど歯擦音の量。潰れて聞こえない／強すぎて刺さる | `crushed`, `too_strong` |
| `consonant_attack` | 破裂音・語頭のアタック（タ・カ・パ、アドリブの頭など）の強さ | `crushed`, `too_strong` |
| `step_change` | ある時点を境に、声の距離・音量・質感が一段変わる（区間切替の段差） | `too_far`, `too_close`, `louder`, `quieter` |
| `buried` | 声がオケに負けて聞き取れない（音量・帯域のマスキング） | `buried` |
| `sits_on_top` | 声がオケの上に載っていて混ざっていない（分離しすぎ） | `not_blended` |
| `thin_light` | 声が軽い・細い・中域が足りない（芯・密度の不足） | `too_light`, `mid_scooped` |
| `muddy` | こもり。中低域が多い、または高域が足りず抜けない | `too_muddy` |
| `airy_fizzy` | エアリーすぎる、高域にジリジリした不要な成分がある | `too_airy`, `fizzy` |
| `space_excess` | リバーブ・ディレイなど空間系の量や質がオケと合っていない | `too_much`, `too_little`, `mismatch` |
| `delay_offbeat` | ディレイ（飛ばし含む）が曲の拍に合っていない | `off_grid` |
| `fx_dropout` | 演出エフェクト（ラジオボイス等）が語尾などで抜けている・途切れる | `missing_fx` |
| `harmony_prominent` | ハモリ（特に生成ハモリ）の主張が強すぎる・生成感が目立つ | `too_strong` |
| `distance` | 声の距離感そのもの | `ok`, `too_far`, `too_close` |

新しい分類が必要になったら、この表と `tests/test_evaluations.py` の `CATEGORIES` を同時に更新してください。

## 指標の検証（Phase 2）

指標の実装は `src/mix_automation/v8/metrics/`（純関数、定数には出典コメント）。手順:

1. `.venv\Scripts\python.exe scripts\analysis\build_reference_stats.py`（既存の分離ステムを使う。中間結果は `data/processed/v8_eval_cache/`、約 30 分）
2. `.venv\Scripts\python.exe scripts\analysis\evaluate_issues.py --write-md`（既定は `data/archive/gemini_v7_20261126` の 4 曲。Beat This! に GPU を使う）

追加の依存（pyproject には未登録。torch を入れ直さないよう `--no-deps`）:
`uv pip install --python .venv\Scripts\python.exe --no-deps beat-this einops soxr rotary-embedding-torch tqdm blind_rt60` と `uv pip install --python .venv\Scripts\python.exe matplotlib`（blind_rt60 が import する）。
