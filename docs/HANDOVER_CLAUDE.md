# Claude への開発引継ぎ（2026-11-26 / v7.1）

このドキュメントは、Gemini（Antigravity）で進めていた開発を **Claude（Claude Code）が引き継いだ** 時点の記録と、ユーザーがこのシステムを最低限使うための手順をまとめたものです。

> **以降の開発は v8 への作り直しとして進めています。** 方針と段階は [V8_PLAN.md](./V8_PLAN.md)、ユーザー評価（正解データ）は [evaluations/](./evaluations/README.md) を参照。v7 の状態はタグ `gemini-v7.0` / `gemini-final` と `data/archive/gemini_v7_20261126/` に固定済みです。

---

## 1. 引継ぎの経緯

- 2026-11-26、Antigravity 上の Gemini が利用制限（Individual quota / AI Credits 不足）に達し、会話「開発経緯の引き継ぎと優先度整理」が作業途中で止まった。
- 止まった時点の Gemini のタスクリスト（7 ステップ中 5 完了）:
  - ✓ Forensic audit of mix defects / Ingest new reference song batch / Fix Studio One song sync / Fix vocal clarity and dynamics / Rebalance harmony and choir buses
  - 未完了: **Run tests and re-mix songs** / **Audit mixes and push commits**
- 直前のユーザーの指摘（08:45）: 「エバの完成マスターでボーカルが埋もれ、声がこもってオケと合わず、サ行やアタックが潰れている。-19.4 dB 下げたままのオフボを入れてしまったので修正版に差し替えた」「シックスフィートアンダーはコーラスがほとんど聞こえず、ハモリはまだ目立つ」「ラブ&デストロイは .song が崩壊していた」
- 最後のユーザー指示（13:23）: 「続きをやって。追加で何か聞かなくてもいいようにして、システムの使い方も分かりやすくしておいて」
- Claude はこのターンの残り（全曲の再レンダー・監査・コミット・使い方の整備）を引き継いで実行した。

## 2. 中断時点の状態（Claude が確認した事実）

| 項目 | 状態 |
| :--- | :--- |
| Git | `main` = `40a85cc`（v7.0）、origin と同期。Gemini の作業はすべて**未コミット**（12 ファイル、+12404 / −5177。大半は 87 曲で再構築した `profiles/pro_target_profiles.json`） |
| Gemini の計画ファイル | `task.md` / `implementation_plan.md` / `walkthrough.md` は存在しなかった（会話ログのみ） |
| 全曲再レンダー（1 回目） | エバ・シックスフィートアンダーは完了。ラブ&デストロイの途中でプロセスが **アクセス違反（0xC0000005）で異常終了** |
| 全曲再レンダー（2 回目） | ラブ&デストロイが 3 回とも `LibsndfileError: Error opening …00_offvocal_raw_backup.wav: System error` で失敗。Fall in Love は完了。benchmark-ab は 2/4 PASS |
| ruff | Gemini は「エラー 0」と報告していたが、それは `--select E,F,W,I` に絞った実行の結果。pyproject の設定どおりだと **129 件**（ほとんどは v7.0 以前からある BLE001 / RUF046 / B008 / S110 など） |
| pytest | 26 件すべて PASS |

## 3. Claude が行った修正（v7.1）

1. **Gemini の未コミット作業をそのままコミット**（内容は無改変。コミットメッセージに Gemini による作業だと明記）
2. **ラブ&デストロイ失敗の原因修正**
   - 原因: Studio One が前回出力の `ラブandデストロイ_AutoMIX.song` を開いたまま（未保存）で、`Media/*.wav` をロックしていた。同じフォルダに上書きしようとして libsndfile が失敗していた。曲固有の問題ではない。
   - 修正: 出力フォルダ内の WAV / .song がロックされていたら、`<曲>_auto_YYYYMMDD-HHMMSS` という新しいフォルダへ書き出す（`pipeline.resolve_writable_output_dir`）。開いている Studio One のセッションには触れない。
3. **mix_report.json に使用素材を記録**（`input_assets`）: 実際に使ったオフボのファイル名・元 LUFS・ステージングゲイン・オフボ候補一覧・各ボーカルステムのファイル名と LUFS・本家音源・自 MIX・歌詞ファイル・出力フォルダ。`MIX_DIRECTOR_REPORT.md` にも表示する。
4. **benchmark-ab の重複監査を解消**: 同じ曲の出力フォルダが複数ある場合（古い `_consensus`、上記の時刻付きフォルダなど）は、最新の `mix_report.json` だけを監査する。
5. テスト 3 件を追加（ロック時の退避フォルダ、オフボ候補の記録と自 MIX / Mixdown の除外、benchmark の最新選択）。全 29 件 PASS。

## 4. 全 4 曲の結果（Claude による再レンダー、2026-11-26 13:55〜14:33）

全 4 曲を最新コードで 1 曲ずつ `mix-auto mix … --style auto` で再レンダーし、`mix-auto benchmark-ab` で監査した。**Do-Not-Regress は 2 / 4 PASS**。

| 曲 | 出力フォルダ | 使用オフボ（元 LUFS） | K-LUFS V/I 実測 / 目標（誤差） | 判定 | FAIL の理由 |
| :--- | :--- | :--- | :--- | :---: | :--- |
| エバ | `エバ_auto` | `…エバoffvocal(修正版).wav`（−16.94） | +2.14 / +1.49（0.65 dB） | PASS | — |
| シックスフィートアンダー | `シックスフィートアンダー_auto` | `シックス・フィート・アンダーINST_….mp3`（−9.28） | +1.08 / +0.87（0.21 dB） | PASS | — |
| ラブ&デストロイ | `ラブandデストロイ_auto_20261126-135529` | `…off vocal_mi8kjomi….wav`（−14.81） | +2.21 / +2.33（0.12 dB） | FAIL | こもり判定: boxiness −3.88 dB（許容上限 −4.02）。300〜600 Hz 付近が目標より約 1.3 dB 多い |
| Fall in Love | `Fall_in_Love_auto` | `20261125 - inst.wav`（−20.21） | +0.91 / +1.93（1.01 dB） | FAIL | ボーカルが目標より約 1 dB 小さい＋暗い（sibilance −22.2 / 目標 −17.4、air −23.8 / 目標 −19.8） |

- ラブ&デストロイの元の出力フォルダ `ラブandデストロイ_auto` は、Studio One で開かれたまま（未保存）だったので一切触っていない。最新の結果は時刻付きフォルダの方。
- エバは修正版オフボが選ばれたことを `input_assets` で確認できた。数値は 11:35 の Gemini レンダーと同じ（パイプラインは決定論的）。
- Fall in Love の FAIL は Gemini の 13:45 レンダーでも同じ値で、Claude の変更（出力先・レポート・監査の選び方）は DSP に影響していない。

## 5. 使い方（ユーザー向け・最低限）

### 5.1 曲フォルダの置き方

`C:\MIX_automation\data\raw\<曲名>\` に素材を入れます。ステムは `stems\` サブフォルダに入れても、曲フォルダ直下に置いても構いません。

| 置くもの | ファイル名のルール（大文字小文字は無視） | 必須 |
| :--- | :--- | :---: |
| ボーカルステム（WAV） | 役割が名前で分かるようにする: `main` / `lead` / `メイン`、`harm` / `ハモ`（`+3` `上ハモ` `upper` = 上ハモ、`-3` `下ハモ` `lower` = 下ハモ）、`choir` / `chorus` / `コーラス`、`double` / `ダブル`、`superoc` / `上オク`、`suboc` / `下オク` | ✓ |
| オフボーカル | 名前に `offvocal` / `off vocal` / `inst` / `オケ` のどれかを含める（WAV 推奨） | ✓ |
| 本家音源（任意） | 上のどれにも当たらない音声ファイル（mp3 など）。Stage ③ の演出検出に使う | |
| 自分の MIX（任意） | 名前に `自mix` を含める。比較用 | |
| 歌詞（任意） | `.txt`。子音・母音アタック保護のアライメントに使う | |

- **オフボを直した場合**: 新しいファイルの名前に `修正` を入れて同じフォルダに置けば、そちらが優先されます（例: `xxx_offvocal(修正版).wav`）。古いファイルは残しても構いません。どちらが使われたかは出力の `mix_report.json` の `input_assets.offvocal_file` で確認できます。
- **書き出し物は名前で除外されます**: `Mixdown` / `AutoMIX` / `_Master` / `_VocalBus` / `オケなし` を含むファイルは素材として読み込まれません。Studio One から書き出した音源を曲フォルダに置いても大丈夫です。
- 日付などの数字に注意: ステム名に `-3`〜`-5` や `+3`〜`+5` が含まれるとハモリと判定されます（例: `2026-03-…`）。ステム名には日付を入れないほうが安全です。

### 5.2 MIX の実行

PowerShell で（曲名は自分のフォルダ名に置き換え）:

```powershell
C:\MIX_automation\.venv\Scripts\mix-auto.exe mix "C:\MIX_automation\data\raw\曲名" --style auto
```

- 1 曲あたり約 10 分（GPU と実機 VST3 を使う）。**同時に 2 曲以上走らせない**でください。
- オプション: `--taste none`（梓川テイストを混ぜない）、`--taste sou --taste-weight 0.30`（別の MIX 師テイスト）など。`mix-auto mix --help` で一覧を表示できます。

### 5.3 出力の場所と Studio One での開き方

`C:\MIX_automation\data\outputs\<曲名>_auto\`（`&` は `and`、空白は `_` に置き換わります）

| ファイル | 内容 |
| :--- | :--- |
| `<曲>_AutoMIX_Master.wav` | 完成 2mix |
| `<曲>_AutoMIX_VocalBus.wav` | ボーカルバスのみ |
| `<曲>_AutoMIX.song` | Studio One 5 プロジェクト（ダブルクリックで開く。プラグイン設定込み） |
| `<曲>_Director_Harmony_Guide.mid` | ハモリ・対旋律の MIDI ガイド（Studio One へドラッグ） |
| `MIX_DIRECTOR_REPORT.md` / `mix_report.json` | 診断レポート（使ったオフボ・ステムもここに記録） |
| `Media\` | `.song` が参照する WAV |

- **Studio One でその曲の `.song` を開いたまま再 MIX すると**、Media がロックされているため `<曲名>_auto_YYYYMMDD-HHMMSS` という別フォルダに出力されます（エラーにはなりません）。最新の結果はその時刻付きフォルダにあります。
- 手直しした `.song` から好みを学習させる場合: `mix-auto learn-preferences --help` を参照。

### 5.4 複数曲の聴き比べ（任意）

```powershell
C:\MIX_automation\.venv\Scripts\mix-auto.exe benchmark-ab
```

`docs\reports\ab_blind_test\index.html` をブラウザで開くと、-14 LUFS に揃えたクリップで聴き比べができます（Do-Not-Regress 監査の結果も表示されます）。

## 6. 既知の問題・気づいた点

### 音質（数値上の懸念。試聴での確認が必要）

1. **ボーカルの幅が極端に狭い**: サビ区間のボーカルバス Side/Mid は、プロ目標が約 −8 dB（エバ・ラブ&デストロイの目標値 −8.2 / −8.3 dB）なのに対し、自動 MIX はエバ −18.9 / Fall in Love −17.1 / ラブ&デストロイ −13.8 / シックスフィートアンダー −10.4 dB。ボーカルがほぼモノラルの細い線になっており、「オケと合わない」「埋もれる」という聴感の一因の可能性がある（推測）。現行の Do-Not-Regress 監査はこの項目をチェックしていない。
2. **エバのサビでボーカル芯が小さい**: サビ（09_Chorus）の帯域全体 V/I は自動 MIX の芯 −0.99 dB に対しプロ目標 +0.76 dB。K-LUFS（聴感重み付き）では目標を上回っているので、「高域を足して大きく聞かせ、中低域の芯が足りない」バランスになっている可能性がある（推測）。
3. **Fall in Love はボーカルが小さく暗い**（上表）。ラブ&デストロイは中低域がやや多い。
4. 2026-11-26 08:45 の「サ行・アタックが潰れる」という指摘に対し、現在の下処理の実測は De-ess の GR 0.65 dB・アタック帯域変化 −0.01 dB で、下処理段では潰していない。潰れが残っている場合は後段（コンプ・Soothe2・Bark シェイパー・マスタリング）が原因の可能性が高い。

### システム・運用

5. **Gemini の報告と実態の食い違い**: 「ruff エラー 0」はチェック項目を絞った実行の結果（フル設定では 129 件、ほとんどは v7.0 以前から存在）。「全 4 曲の再レンダーと benchmark-ab を実行中」と報告していた後、実際にはラブ&デストロイが異常終了・ロックエラーで一度も完了していなかった。
6. **Studio One で開いている曲を再 MIX すると別フォルダに出力される**（v7.1 で対処。上の 5.3 参照）。不要になった時刻付きフォルダは手動で消して構わない。
7. `ラブ&デストロイ\stems\test20261124 - main 1.wav` は無音（LUFS 計測不能）で、自動的に使われていない。`main1.wav` とは別ファイル。
8. オフボ候補の選び方（`pipeline.discover_song_assets`）の細かい癖:
   - 名前に `rev` や `master` を含むファイルも「修正版」扱いで優先される。一方で `_master` を含むファイルは書き出し物としてそもそも除外される。
   - 本家音源は「どの規則にも当たらない最後の音声ファイル」が選ばれるので、関係ない音声ファイルを曲フォルダに置くと誤認される可能性がある。
9. ステム名の `-3`〜`-5` / `+3`〜`+5` は日付の一部でもハモリと判定される（5.1 参照）。
10. Gemini の 1 回目の全曲レンダーは VST3 のネイティブクラッシュ（0xC0000005）で落ちていた。Gemini が Nectar 3 / VocalSynth 2 をサブプロセスに分けた後の今回の 4 曲連続レンダーでは再発しなかったが、根本原因は未確認。
11. GEMINI.md・README の一部の数値（テスト件数「25」「21」、リファレンス「44 曲」）は古い。現在はテスト 29 件、リファレンス 87 曲。
12. **`<曲>_AutoMIX_VocalBus.wav` は 2mix に入った音量と一致しないことがある**: ピークが 0.89 を超えると書き出し時に正規化される（`pipeline.py` の `vocal_bus_export`）。シックスフィートアンダー（2026-11-26 レンダー）はマスター内より −5.3 dB 小さい（マスターをブロックごとに `a·inst + b·vocalbus` で当てはめた b/a の中央値 1.85）。解析では `scripts/analysis/_v8eval.py` の `vocal_bus_gain` で補正している。v8 では書き出しで正規化しない。
13. **`Media/02_mainvocal_prepared.wav` は演出区間が抜かれている**: Stage ③ のインライン演出（ラジオボイス・オートチューン等）の区間は Studio One 用に `12_fx_*`／`13_fx_*` へ切り出され、prepared 側はその区間が無音（`inline_duck_mask`）。処理前の声が必要なら prepared ＋ 演出トラックを足し戻す（`_v8eval.true_pre_vocal`）。古いレンダーの `14_fx_distortion_trash2.wav` が残っていることがあるので、mix_report の演出イベントに無い種類は足さない。
14. **評価用の追加パッケージ**（`pyproject.toml` の `eval` extra、uv.lock も更新済み）: torch を入れ直さないよう `uv pip install --python .venv\Scripts\python.exe --no-deps beat-this einops soxr rotary-embedding-torch tqdm blind_rt60` と `uv pip install --python .venv\Scripts\python.exe matplotlib faster-whisper pystoi` で入れた。faster-whisper は CPU 版 `onnxruntime` を引き込み、既存の `onnxruntime-gpu` と同じフォルダを共有する。GPU の ONNX 推論がおかしくなったら `uv pip install --python .venv\Scripts\python.exe --reinstall onnxruntime-gpu==1.32.0` で入れ直す。なお uv.lock の torch（2.15.0）と .venv の torch（2.11.0+cu128）は以前から一致していない（`uv sync` すると torch が入れ替わるので注意）。

### 次にやるとよいこと（優先順）

1. 4 曲の新しいマスターを試聴して、上記 1〜3 の数値上の懸念が聴感と一致するか確認する（特にエバのボーカル幅と芯）。
2. 一致するなら、ボーカル幅（Side/Mid）を目標プロファイルに含めて最適化と Do-Not-Regress 監査の両方に加える（個別曲の対症療法ではなく、全曲に効く構造的な対処として）。
3. Fall in Love の K-LUFS 誤差と暗さの原因（目標合成か、最適化の制約か）を調べる。
