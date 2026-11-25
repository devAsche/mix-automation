# YouTube References Inbox (`C:\MIX_automation\youtube_references`)

`muse` エージェント等で収集した新規リファレンス楽曲セット（歌ってみたカバー ＋ 公式オフボーカル ＋ ボカロ／原曲本家）を配置するステージングディレクトリです（音源類は `.gitignore` 対象）。

本ディレクトリの構造は既存の `C:\MIX_automation\data\references`（44 曲収録済み）と完全互換になっており、品質検品ゲートを通過した楽曲から `profiles/pro_target_profiles.json` の構築パイプラインへ統合されます。

## 1. ディレクトリ構成・ファイル命名規則

1 曲につき `<Group>\<NN_Title_Artist>\` の 1 フォルダを作成して格納してください（`<Group>` は歌い手名または MIX 師名の英字識別子、例: `Azusagawa`, `Sou`, `TOY`, `Saburou` など）。

```text
C:\MIX_automation\youtube_references\
├── _info\
│   ├── existing_44_songs.json      # 既存収録済み44曲の重複防止インデックス（自動生成済み）
│   └── manifest_batch4.json        # 新規収集分のメタデータJSON（batch番号や日付で追記・分割可）
└── <Group>\
    └── <NN_RomajiTitle_Producer>\
        ├── cover_<CoverYouTubeID>.webm       # 1. 歌ってみた音源（YouTube最高音質 Opus 251 / m4a / 高音質MP3等、再エンコードなし推奨）
        ├── original_<OriginalYouTubeID>.webm # 3. ボカロ／原曲本家音源（同上）
        └── offvocal\                         # 2. 公式配布オフボーカル一式
            ├── <公式配布オフボ>.wav / .mp3    # piapro / Google Drive / Dropbox / 公式配布サイトの原ファイル
            └── ...                           # キー違い音源や歌詞txt・UST等があれば同梱可（MV動画や巨大zip本体は展開後に削除）
```

## 2. ローカルで利用可能なツール

- `yt-dlp`: `C:\MIX_automation\tools\yt-dlp.exe`
- 重複確認用リスト: `C:\MIX_automation\youtube_references\_info\existing_44_songs.json`（収録済み 44 曲の `cover` / `original` YouTube 動画 ID 一覧）

## 3. `_info\manifest_*.json` のレコード形式

`C:\MIX_automation\src\mix_automation\analysis\reference_extractor.py` がそのまま読み込めるよう、以下の JSON 配列形式で記録してください。

```json
[
  {
    "group": "Sou",
    "vocal": "Sou",
    "mixer": "Sou (Vocal&Mix)",
    "artist": "ボカロP・原曲作者名",
    "title": "曲名",
    "genre_hint": "ballad | sparse_pop | pop_ballad | mid_pop | dense_rock | edm",
    "cover": "歌ってみたのYouTube動画ID（11文字）",
    "original": "本家のYouTube動画ID（11文字）",
    "offvocal_source": "配布元URL（piapro / Google Drive / Dropbox 等）",
    "offvocal_note": "配布形式メモ（例: Drive配布 inst WAV / piapro公式MP3 Mst+noMst 等）",
    "dir": "Sou\\03_SongTitle_Artist"
  }
]
```

## 4. 収集時の優先事項・注意点

1. **重複回避**: `_info\existing_44_songs.json` に含まれる `cover` 動画 ID と同じ動画はスキップしてください。
2. **音質維持**: YouTube 音源は `-f bestaudio`（Opus 251 `.webm` や AAC `.m4a`）のまま無劣化保存するか、変換する場合も高ビットレートを維持してください。
3. **オフボーカルの純度**: 公式リンク（piapro、Google Drive、Dropbox、公式サイト等）から正規オフボを取得し、アーカイブ展開後は元の `.zip` や不要な MV 動画（`.mp4` / `.mov`）を削除して容量を節約してください。ボーカル抜き AI で作った非公式ニコカラ等は原則避け、やむを得ない場合は `offvocal_note` に必ず明記してください。
