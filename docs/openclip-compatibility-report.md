# OpenCLIP互換性・Zero-shot統合レポート

## 対象と評価条件

- ライブラリ: `openclip-gemma` / `openclip_gemma`
- モデル: `google/embeddinggemma-2`
- 固定revision: `914f7f89142e33e77833254d9c9b90c3cef7303b`
- 評価環境: Python 3.12.10、PyTorch 2.14.1+cpu、Transformers 5.20.1、Sentence Transformers 6.1.0
- CPU FP32を主基準とし、CPU BF16も別途測定。CUDAは利用できないため未検証。
- M4ではOpenCLIP APIの呼び出し形状と代表的な推論フローを評価した。全OpenCLIPモデル・全入力との互換性を意味しない。

数値比較の暫定FP32基準は、各埋め込みの最大絶対誤差 `<= 1e-4`、コサイン類似度 `>= 0.9999`。BF16にはこのFP32基準を適用しない。

## API互換性一覧

| API / 挙動 | 判定 | 評価 |
|---|---|---|
| `create_model()` | 部分互換 | Factoryとしての役割は同じだが、`embeddinggemma-2`と固定されたEmbeddingGemma 2 checkpointのみ対応。`pretrained=None`、未対応モデルや引数は拒否する。 |
| `create_model_and_transforms()` | 部分互換 | `(model, preprocess_train, preprocess_val)`の3要素を返す。学習augmentationはなく、train/val両方で決定的な評価前処理を返す。前処理と画像形状の契約はOpenCLIPモデルと異なる。 |
| `get_tokenizer()` | 部分互換 | 文字列または文字列リストを受け取り、呼び出し可能なTokenizerが`torch.LongTensor[B,L]`を返す。Gemmaの語彙・特殊トークンを使用するため、OpenCLIPのCLIP token IDとの互換性はない。Paddingは動的であり、OpenCLIPの既定context length 77の固定幅paddingとは異なる。 |
| `model.encode_image()` | 部分互換 | `[B,3,H,W]`から`[B,768]`を返し、バッチ順を保持する。入力TensorはこのライブラリのEmbeddingGemma用前処理を経た値である必要があり、OpenCLIP向けに前処理したTensorをそのまま使えるとは限らない。異なる高さ・幅の画像は単一のdense Tensorに`stack`できない。 |
| `model.encode_text()` | 部分互換 | `[B,L]`のGemma token IDから`[B,768]`を返し、公式のprompt・special token・right-padding規約を使う。CLIP token IDやCLIP固定長系列はそのまま利用できない。 |
| `model.eval()` | 互換 | 通常の`torch.nn.Module`の評価モード切替を利用する。 |
| `model.to(device)` | 部分互換 | PyTorch Moduleとしてdevice移動する。FP32/BF16を対象とし、FP16変換は明示的に拒否する。 |
| `forward()` / `get_logits()` | 非対応 | 意図して未実装。画像・テキストのスコアは利用側で埋め込みから計算する。 |
| OpenCLIPの学習・fine-tuning API | 非対応 | v0.1.0の対象外。 |

### 互換性を保つ際の注意

- `normalize=False`でも、EmbeddingGemma/Sentence Transformersのネイティブ出力が正規化済みであれば、その出力を返す。OpenCLIP側の「未正規化特徴量」と同じ意味ではない。
- EmbeddingGemma公式タスクpromptはTokenizer生成時に明示する。CLIP専用promptや学習済み`logit_scale`は作らない。
- OpenCLIPの典型例はコサイン類似度を`100.0`倍してsoftmaxへ渡すが、このライブラリはその係数を模倣しない。例のスコアは非較正のcosine similarity。任意表示のsoftmaxも候補間の相対値であり、分類確率ではない。
- 異なる形状の画像をまとめる未検証のリサイズやpaddingは行わない。同一shapeの前処理済み画像をバッチ化する。

## Zero-shot統合検証

`examples/zero_shot_classification.py`を2枚のデモ画像と英語・日本語を含む3候補で実行した。CPU FP32でcosine similarity行列を得て、オプションの相対softmaxも表示した。

| 画像 | 赤い円 / 青背景 | 緑の三角形 / 黄背景 | 日本語候補 |
|---|---:|---:|---:|
| デモ画像1 | 0.767742 | 0.632870 | 0.743515 |
| デモ画像2 | 0.631371 | 0.793333 | 0.612163 |

スコア行列は`[2,3]`で、有限値を確認した。相対softmaxは各画像の候補スコア間で計算し、分類確率とは表示していない。UTF-8出力を有効にした実行で、日本語候補ラベルも正しく表示された。

実モデル統合テストでは固定revisionで英語・日本語候補、1画像/複数画像、候補順・画像順の入れ替え、バッチサイズ差を確認した。画像・テキスト埋め込みの次元は768であり、画像と候補テキストから作る行列は`[画像数, 候補数]`。推論呼び出し間でモデルやProcessorが再ロードされないことも確認した。

### Sentence Transformers公式経路との比較

CPU FP32では、同一revision・同一入力・同一dtype/deviceで比較し、以下の最大絶対誤差を得た。各コサイン類似度は丸め表示で約`1.0`。

| 比較対象 | 最大絶対誤差 | 判定 |
|---|---:|---|
| 画像埋め込み | `2.7381e-7` | 基準内 |
| テキスト埋め込み | `2.8075e-6` | 基準内 |
| cosine similarity行列 | `1.1921e-6` | 基準内 |

CPU BF16では同じ統合比較で画像埋め込み、テキスト埋め込み、スコア行列の最大絶対誤差はいずれも`0.0`、コサイン類似度は約`1.0`だった。これは当該環境・revisionでの観測であり、全デバイスでのBF16一致保証ではない。CUDAは利用不可のため未検証。

Zero-shot統合テストは**2 passed, 1 skipped**（CUDAケースをskip）。単体テストは当該時点で**74 passed**。

## Tokenizerのpaddingとcontext length

| 項目 | 根拠・評価 |
|---|---|
| OpenCLIP Tokenizer | 公式実装の既定context lengthは77で、固定長にpaddingする。 |
| 本ライブラリ | 動的paddingを使い、各バッチの最長系列までを返す。OpenCLIPの固定幅77を前提とする呼び出し側とは完全互換でない。`context_length`を指定した場合もサポート上限を超えられない。 |
| モデルカード上の長さ | EmbeddingGemma 2の公式モデルカードは共有context windowを8192 tokensとしている。 |
| config値 | 固定revisionの`max_position_embeddings`は262144。config値のみから262144 tokensの実用性・公式サポートを推定しない。 |
| Sentence Transformers値 | `max_seq_length`は非常に大きいsentinel値で、モデルカードの明示的なcontext windowに代わる上限保証ではない。 |
| 採用上限 | ライブラリは`min(8192, max_position_embeddings)`を上限とする。通常の短文バッチを8192までpaddingする設定ではない。 |
| メモリ | 8192長入力でのピークメモリや最大バッチサイズは測定していない。8192はモデルカードに基づく最大系列長であり、実用上のメモリ保証ではない。必要に応じて短い`context_length`と小さなバッチを指定する。 |

現時点では動的paddingを維持する。CLIP互換の固定長paddingを既定にすると、Gemmaで不要な計算を増やし、公式Tokenizer契約とも異なるため採用しない。

## 実行例

```powershell
uv run python examples\zero_shot_classification.py --demo --show-relative-softmax --device cpu
```

実行例は未スケールcosine similarityと、明示指定時のみ相対softmaxを表示する。実画像を指定する場合、1つの呼び出しで処理する画像は前処理後の高さ・幅が一致する必要がある。

## 公開前の残課題

1. CUDAでのFP32/BF16モデルロード、画像・テキスト推論、精度とメモリを検証する。現評価環境はCPU専用。
2. 8192 tokens近傍の実用的なメモリ・レイテンシ・バッチ上限を、対象ハードウェアごとに計測する。
3. 公開するPyTorch/Transformers/Sentence Transformersバージョン範囲とOS/PythonのCI matrixを定め、clean installとwheel利用者経路を検証する。
4. 入力画像のshape制約、Gemma用前処理、動的padding、Gemma token ID、`normalize=False`のネイティブ出力意味をAPIドキュメントで目立つ形で告知する。
5. `forward()` / `get_logits()`、CLIPのlogit scale、学習APIは依然非対応。これらを前提とする既存コードは変更なしでは動作しない。
6. BF16の誤差・可用性は環境依存であり、FP32の暫定誤差基準をBF16へ流用しない。

## 参照資料

- [OpenCLIP README / inference example](https://github.com/mlfoundations/open_clip/blob/main/README.md)
- [OpenCLIP Tokenizer source](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/tokenizer.py)
- [OpenCLIP model source](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py)
- [OpenCLIP factory source](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/factory.py)
- [EmbeddingGemma 2 model card](https://huggingface.co/google/embeddinggemma-2)
- [EmbeddingGemma 2 pinned config](https://huggingface.co/google/embeddinggemma-2/blob/914f7f89142e33e77833254d9c9b90c3cef7303b/config.json)
- [Project compatibility specification](./compatibility-spec-v0.1.0.md)
- [M0/M0.1 validation record](./m0-validation.md)
