# MIX 知識（v8 再設計 R1、2026-11-27）

- [rulebook.md](rulebook.md): 解説動画 10 本から作った処理の骨格と数値（出典は `ID@分:秒`）
- [rules.yaml](rules.yaml): 同じ内容を機械で読める形にしたもの（22 段）
- [plugin_map.md](plugin_map.md): 手法と手持ち VST3 の対応表、および空白
- [gap_vs_v7.md](gap_vs_v7.md): v7 のチェーンとの差分

スクリプトは `scripts/knowledge/` にあります。キャッシュは `data/processed/mix_knowledge/` に置いていて、コミットしません。
- `inventory_amo.py`: Amo 15 曲の棚卸し
- `match_practice_cover.py`: practice と cover の照合
- `transcribe_videos.py`: 文字起こし（faster-whisper large-v3、GPU）
- `frames_ocr.py`: 場面転換の検出、フレーム選び、OCR（EasyOCR は `C:\MIX_automation_ext\ocr\pylibs`）
- `audit_extra_plugins.py`: 未監査の VST3 の読み込みテスト
- `extraction_counts.py`: 動画ごとの抽出件数

## 素材の棚卸し（Amo_*、15 曲）

- cover と original は全曲にあり、長さの差は 4 秒以内です。
- **inst は 15 曲とも無し**です。
- 「照合」は、補正後メインと全 cover の、キーを考慮したクロマ相関で判定しました（z 値で正規化）。

| 曲 | cover / orig | practice（補正前 → 後） | practice 長 | 照合 |
|---|---|---|---|---|
| ApocalypseTokyo | 3:17 / 3:16 | M1 H2 → 同 | 3:22 | OK |
| Bibibibi | 4:01 / 4:01 | M1 H2 Oct1 → 同 | 4:03 | OK |
| CatLoving | 1:56 / 1:56 | M4 H4 → M2 H4 Dbl2 | 1:57 | OK（弱い） |
| DayByDays | 2:09 / 2:09 | 2 人: M4 H7 Oct2 → 同 | 2:10 | OK |
| Disbarhythm | 2:55 / 2:55 | M1 H3 Oct1 → 同（+MIDI ガイド） | 2:57 | OK |
| Dokuzu | 1:59 / 1:58 | M2 H5 Oct1 セリフ2 → 同 | 2:35 | ✗ 長さが合わない |
| Henceforth | 3:56 / 3:55 | マイク / IF 比較のクリップ（約 0:30）だけ | 0:34 | ✗ 歌のテイクが無い |
| Identity | 2:35 / 2:35 | ファイル名が「侵蝕」 | 2:52 | ✗ Shinshoku の cover と一致 |
| KimiNoMyaku | 4:04 / 4:00 | ファイル名が「オカルトガール」 | 3:25 | ✗ 該当する cover なし |
| Kyuukurarin | 3:35 / 3:36 | M1 Dbl1 → 同（+生成ハモ2） | 3:37 | OK |
| LamelteonsNight | 3:32 / 3:31 | M2 H4 Oct1 → M1 H2 Oct1 | 4:04 | ✗ 別の曲 |
| QUEEN | 2:21 / 2:22 | M1 H3 Oct1 → M1 H6 Oct1 | 2:22 | OK |
| Requiem | 2:37 / 2:37 | M1 サブ2 H4 Oct4 → 同 | 3:31 | ✗ LamelteonsNight の cover と一致 |
| Shinshoku | 2:53 / 2:50 | M1 Oct1 → 同 | 4:02 | ✗ 別の曲 |
| StardustMedley | 2:20 / 2:20 | M2 H6 Oct1 → 同 | 2:28 | OK |

## 動画ごとの抽出件数

| ID | 長さ | 発話 | フレーム | OCR 設定値の枠 | ノート処理行 | 数値付き行 |
|---|---|---|---|---|---|---|
| -c0bHeO5ISs | 37分 | 543 | 30 | 50 | 12 | 7 |
| 2afvdjOpVJQ | 24分 | 354 | 30 | 19 | 16 | 8 |
| 4-GZf4N1cso | 10分 | 172 | 15 | 0 | 1 | 0 |
| 5NQ7idc9fY8 | 6分 | 9（テロップ 149 行） | 14（+155 枚） | 8 | 7 | 4 |
| E8zUKz21S9c | 34分 | 430 | 30 | 72 | 15 | 7 |
| heG8RtaRwN0 | 20分 | 257 | 24 | 20 | 15 | 5 |
| JUXJShxfzKQ | 54分 | 588 | 30 | 105 | 30 | 11 |
| Mo_qfyNdYQQ | 13分 | 170 | 22 | 2 | 12 | 3 |
| RTVPdDmWrwo | 64分 | 817 | 30 | 24 | 43 | 17 |
| v6-VoBbwsio | 78分 | 1077 | 15 | 0 | 8 | 0 |

OCR で取れたのはプラグイン名が中心で、つまみの値は少なめでした。数値の大半は発話から取っています。
