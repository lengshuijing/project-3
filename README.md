# 神経進化プロジェクト

本リポジトリでは、遺伝的アルゴリズムを用いてニューラルネットワーク（NN）の構造やパラメータを進化させる「**NeuroEvolution**」をテーマにした個人研究をまとめています。具体的には、「NEAT（NeuroEvolution of Augmenting Topologies）」によるエージェント学習と、Backpropagationを組み合わせた進化手法（Backprop NEAT）を検証しました。

## 1. プロジェクト概要

1. **Part 1**: **NEAT を用いた Neural Slime Volleyball 学習**  
   - **目的**: Slime Volleyball 環境でAIと対戦し、NeuroEvolutionでより優れたエージェントを育成  
   - **ポイント**:  
     - 遺伝的アルゴリズムによるネットワーク構造の拡張（ノード・接続の追加）  
     - 最適化されたNNがゲーム内の戦略（ジャンプ・移動等）を自力で獲得  

2. **Part 2**: **Backprop NEAT を活用した分類タスク**  
   - **目的**: 円形データの2D分類において、構造進化 + Backpropを組み合わせた手法で高精度を目指す  
   - **ポイント**:  
     - NEATのトップロジー進化 + Adam オプティマイザを併用することで、構造と重みを同時最適化  
     - 非線形分離が必要なデータセットに対して約95%以上の精度を達成

## 2. ファイル構成

- **Report-Project-Neuroevolution.pdf**  
  NEATのアルゴリズム、Slime Volleyball環境での適用、Backprop NEATによる分類実験など、詳しい実験内容と結果を記載  
- **first.py**  
  NEATアルゴリズム実装やSlime Volleyball環境への適用コード  
- **second.py**  
  Backprop NEATによるネットワーク学習＆分類タスクのサンプル実装  

## 3. 実行環境 / 使用技術

- **Python 3.8+**  
- **gym** (Slime Volleyball環境など)  
- **numpy / jax / PyTorch** (NN実装や学習で併用)  
- **graphviz** (ネットワーク構造の可視化)  
- **imageio** (プレイ画面のGIF化等)

> GPU環境（CUDA）があると学習が高速化されますが、CPUでも小規模テストは可能です。

## 4. 実行方法

1. **環境構築**  
   - Python仮想環境を作成して必要ライブラリを `pip install`  
   - Slime Volleyball 環境:  
     ```bash
     pip install slimevolleygym
     ```
   - Graphviz, imageio 等も必要に応じてインストール  

2. **NEAT学習（`first.py`）**  
   - Slime Volleyballでエージェントを進化的に学習  
   ```bash
   python first.py
   ```
   - 学習完了後、最良エージェントのネットワーク構造を可視化 (network.png生成)

3. **Backprop NEAT（`second.py`）**  
   - 円形データの分類タスクにおけるネットワーク進化＋Backprop実装  
   ```bash
   python second.py
   ```
   - 学習時にLossや精度がログ出力され、最終的なテスト精度を表示

## 5. 実験ハイライト

### Part 1: NEATでのSlime Volleyball対戦

**進化過程:**
- 初期：層や接続の少ないシンプルなNN
- 世代を重ねるごとに隠れノードが増え、戦略が高度化

**成果物:**
- 最終的に内蔵AIを上回るパフォーマンスを獲得
- 作成したGIFでジャンプタイミング・ボールへの対応が改善されている様子を確認

### Part 2: Backprop NEATによる2D分類

**進化 + 勾配降下:**
- NEAT でノードや層を追加しつつ、バックプロパゲーションで重みを更新
- 非線形データ（円形の内外）を約95%以上の精度で分類

**学習曲線:**
- 世代が進むに連れLossが安定して低下
- 複雑な構造が必要な境界も正しく学習できる

## 6. 今後の展望

1. **別タスクへの適用**
   - Atariゲームや連続制御タスクなど、より複雑な環境でNeuroEvolutionの有効性を検証

2. **パラメータ/ハイパーパラメータの最適化**
   - NEATの突然変異率やバックプロパゲーションの学習率を体系的に探る

3. **大規模実験**
   - 大きなネットワーク、より多様な遺伝演算を導入し、解探索の拡張性を確認

## 7. ライセンス / 注意事項

- 本プロジェクトは個人的な研究および学習目的で公開しています。
- MIT License を適用予定ですが、二次利用の際は元となるライブラリ（gym、slimevolleygym等）およびデータセットのライセンスに留意してください。
- 大規模な実行や商用利用への対応は自己責任でお願いします。

