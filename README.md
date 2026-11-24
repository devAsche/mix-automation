# MIX Automation (`mix-automation`)

歌ってみた制作における**「どの楽曲でも破綻しない上級者クオリティの混ざり具合」**と**「高頻度投稿のための圧倒的な時短」**を両立する、Studio One 5.5 ネイティブ対応の専用自動MIXエンジンです。

## 主な機能（目標①〜④ 一貫対応）

1. **上級者リファレンス15曲に基づく「対オケ相対音響プロファイル」最適化（目標①）**
   梓川氏・TOY氏・さぶろう氏の計15曲の完成音源をGPU音源分離して抽出した「対オケ6帯域マージン」「対オケ・クレストファクター比（張り付き度）」「短期位置安定度」「2〜5kHz共振指数」をターゲット関数とし、曲調やオケの音圧が変わっても破綻しないプラグイン設定値をCMA-ES最適化で逆算します。
2. **パート間の音圧変化に自動追従する「アンカー伝播（Anchor Propagation）」**
   Aメロ・Bメロ・落ちサビ・サビなどの構造を自動分割し、基準サビ（アンカー）の混ざり具合を他セクションのオケ密度差に合わせて各イベントのクリップゲインへ自動展開します。Studio One上で1パートだけ微調整した後、`propagate-anchor` コマンドで全パートへ再反映することも可能です。
3. **iZotope RX 10 下処理・音素連動ゲインライディング・Soothe2 ポリッシュ（目標②）**
   実機の `RX 10`（Mouth De-click / De-plosive / Voice De-noise / Breath Control / De-ess）と `Oeksound Soothe2`、`Exponential Audio R2`、`Wave Breaker` 等を直接駆動し、無音ストリップ・ニュアンス温存ピッチ安定化・母音コア連動プリゲイン均しからマスタリング（`-8.2 LUFS`）までを一気通貫で処理します。
4. **フレーズ語尾の儚いリバーブ投げ自動生成（目標③）＆ ハモリ提案ガイド生成（目標④）**
   余白のあるフレーズ語尾を検出してロングリバーブへ飛ばす `fx_ethereal_throws` トラックと、オケのキー・スケールを解析してあなた自身の声で歌わせた3度上・3度下ハモリガイドトラック（Studio One上でミュート状態で同梱）およびディレクター構成案レポートを自動出力します。

## 使い方（CLI）

```powershell
# 1. フル自動MIXの実行（Studio One 5 .song + 完成Master WAV + 診断レポート出力）
# スタイル選択: --style consensus (3名統合標準) | azusagawa | toy | saburou
C:\MIX_automation\.venv\Scripts\mix-auto.exe mix "C:\MIX_automation\data\raw\シックスフィートアンダー" --style consensus

# 2. Studio One 5 で1つのパート（アンカー）のクリップゲインを手直しした後、全パートへ同じ混ざり具合を再伝播
C:\MIX_automation\.venv\Scripts\mix-auto.exe propagate-anchor "C:\MIX_automation\data\outputs\...\AutoMIX.song" --anchor 2

# 3. 新しいリファレンス音源を追加した際のプロファイル再構築（GPU実行）
C:\MIX_automation\.venv\Scripts\mix-auto.exe build-references

# 4. ローカルのStudio One 5・VST3プラグイン稼働診断
C:\MIX_automation\.venv\Scripts\mix-auto.exe doctor
```

詳細な内部設計は [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) を参照してください。
