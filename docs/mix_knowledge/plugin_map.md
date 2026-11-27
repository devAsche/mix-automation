# 手法 → 手持ちの実機プラグイン対応表（R1, 2026-11-27）

読み込み確認の出典:
- `data/outputs/vst3_audit_report.json`（`scripts/audit_vst3_load.py`、既存）
- `scripts/knowledge/audit_extra_plugins.py` → `data/processed/mix_knowledge/vst3_extra_audit.json`（今回の追加分）

状態の記号: ◎ = 登録済み・v7 で使用中 / ○ = 読み込めて音が出る（未登録） / △ = 条件付き / × = 読み込めない

| 手法（rulebook の段） | 第一候補 | 他の候補 | 状態・注意 |
|---|---|---|---|
| ノイズ・リップ・ブレス・破裂音 | RX 10 Mouth De-click / De-plosive / Voice De-noise / Breath Control | RX 10 Spectral De-noise, De-reverb | ◎ オフライン専用（v7 で使用中） |
| ピッチ補正 | （素材は補正済み） | MAutoPitch, Waves Tune LT Stereo, Melodyne(ARA) | ◎/○。Waves Tune LT Mono は 2ch 不可 |
| 引き算 EQ（ローカット・こもり・共振） | Persistent Q | ZL Spectrum EQ（内側バイナリ）, MEqualizer, kHs 3-Band EQ / Filter | ◎。ZL は bundle root だと scan_error |
| 共振の動的抑制 | soothe2 | Initial Dynamic EQ | ◎ |
| ディエッサー | RX 10 De-ess | （Pro-DS 系は無し）Initial Dynamic EQ の高域バンド | ◎ |
| 速いピーク用コンプ（1176 系の役） | Softube VCA Compressor | MCompressor, kHs Compressor, kHs Dynamics, Flash(?) | ◎/○。1176 実機エミュは無し |
| 遅い平均化コンプ（LA-2A 系の役） | **UAD LA-2A** | MCompressor（opto 風設定）, TALLBOY(?) | × UAD は scan_error（UAD ランタイムが必要）。v7 は代替で動いているか要確認 |
| 音色 EQ（空気感・胸） | TDR VOS SlickEQ | **UAD Pultec EQP-1A/MEQ-5**, MEqualizer, VPRE-73 | Pultec は × scan_error |
| サチュレーション / 倍音 | Softube Saturation Knob | iZotope Trash 2, MSaturator, MWaveShaper, kHs Distortion, VPRE-73（プリアンプ色付け）, Dagon(?) | ◎/○。VPRE-73/Dagon は内側バイナリ指定で読込可 |
| エキサイター / 明瞭化 | iZotope VEA | Nectar 3（Dimension/Exciter）, LANDR FX Voice | ◎/○。Nectar 3 は読込に 60 秒超かかることがある |
| ダブラー | iZotope Vocal Doubler | kHs Chorus / Ensemble, kHs Pitch Shifter（±数 cent + 短ディレイで自作）, MDelay | ◎ だが v7 は 2mix に出していない（gap_vs_v7 参照） |
| イメージャー / 広がり | **登録なし** | MStereoExpander, kHs Stereo, kHs Haas, Ozone Elements の Imager | ○ 未登録。空白 |
| ショートディレイ（スラップ） | kHs Dual Delay | kHs Delay, MDelay | ◎/○ |
| 音価ディレイ（1/8・1/4・付点） | kHs Dual Delay | MDelay（フィルタ・ダッキング付き）, kHs Delay | ◎/○。ダッキング付きは MDelay のみ |
| プレートリバーブ | Arturia Rev PLATE-140 | MCharmVerb, kHs Reverb | ◎ |
| ルーム / ホール | Exponential R2, PhoenixVerb | MConvolutionEZ, MCharmVerb, kHs Reverb | ◎ |
| センド後の EQ / ダッキング | Persistent Q / MEqualizer + MCompressor（サイドチェイン要確認） | kHs Filter | pedalboard でサイドチェイン入力は使えないため、自前 DSP でエンベロープ追従が現実的 |
| オケ側 EQ（ボーカル帯域を空ける） | 自前の動的ディップ（v7） | soothe2（外部サイドチェイン不可）, Initial Dynamic EQ, MEqualizer | サイドチェインは自前 DSP。M/S 処理は自前 or MChannelMatrix |
| ボーカルライダー / 音量追従 | 自前（v7 の母音ライダー等） | MAGC（自動ゲイン） | ◎/○。Waves Vocal Rider は無し |
| バスのまとめ（グルーコンプ） | Softube VCA Compressor | MCompressor, kHs Compressor | ◎ |
| マスター（EQ・リミッター） | Ozone 9 Elements + Wave Breaker | Ozone 12 Elements, kHs Limiter, kHs Clipper | Wave Breaker は単体 scan だと ×（audit）だが v7 は使用中 → 読み込み経路を要確認。Ozone 12 Elements は param=0（パラメータ非公開） |
| 解析 | SPAN | MAnalyzer, MLoudnessAnalyzer | 表示用で処理には不要 |

## 空白・要対応（R2 以降で判断）

1. **1176 系・LA-2A 系の実機エミュが使えない。** UAD は全部 scan_error。解説では LA-2A / 1176 / Pultec の名前がよく出るため、役割で置き換える（VCA Compressor = 速い、MCompressor の遅い設定 = opto 風）。
2. **イメージャーが未登録。** MStereoExpander / kHs Stereo / kHs Haas が読み込めるので、どれか 1 つを登録する。
3. **ダブラーの音が 2mix に出ていない（v7）。** Vocal Doubler は読み込めるので、配線の問題。
4. **ダッキング付きディレイ / サイドチェイン。** pedalboard の制約で外部サイドチェインは不可。MDelay の内部ダッキングか、自前のエンベロープでセンドを絞る。
5. **Waves はシェル経由で Tune LT しか読み込めない**（14.12/14.17/14.21 はダイアログ待ちでタイムアウト）。解説に出る Waves 製品（H-Delay, L1, MV2, CLA 系, R-Vox, Renaissance）は使えない前提で置き換える。
6. FabFilter（Pro-Q / Pro-C / Pro-DS / Pro-R / Saturn）は未インストール。
