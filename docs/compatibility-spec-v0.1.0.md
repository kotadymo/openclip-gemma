# EmbeddingGemma OpenCLIP 互換性仕様書

- **文書ID**: EGOC-SPEC-001
- **対象リリース**: v0.1.0
- **状態**: Draft / M0〜M3実装完了。M3のCPU FP32/BF16統合検証済み。CUDA未検証
- **作成日**: 2026-10-17
- **配布パッケージ**: `openclip-gemma`
- **対象モデル**: `google/embeddinggemma-2`

## 1. 目的・非目的

### 1.1 目的
既存のOpenCLIP画像・テキスト推論コードに対し、主にimport文とモデル名を変更することでEmbeddingGemma 2による推論を実行できる独立Pythonパッケージを提供する。

### 1.2 優先順位
1. EmbeddingGemma 2公式バックエンドに対する数値的忠実性
2. 呼び出し互換性（関数、主要引数、Tensor形状）
3. エラーの明示性・再現性・保守性
4. 性能最適化

API互換性と数値的忠実性が衝突した場合は後者を優先し、非互換性を明示する。

### 1.3 非目的
OpenCLIPの全モデル対応、OpenCLIP学習基盤、損失関数の完全互換、独自モデル重みの再実装、音声・動画、OpenCLIPと同一の分類確率を保証すること。

## 2. 適用範囲と用語

- **互換**：同じ用途の代表的コードに対し、引数・戻り値・shapeが同じ。
- **条件付き互換**：見かけのI/Oは同じだが意味・数値・制限が異なる。
- **非対応**：`NotImplementedError`または`ValueError`等で明示的に拒否。
- **参照バックエンド**：公式モデルIDに対応するSentence Transformers / Transformers実装。
- **公式一致**：同一重み、dtype、前処理、プロンプト設定による出力が許容誤差内で一致。

## 3. サポート構成

| 項目 | v0.1.0 |
| --- | --- |
| Python | 3.12（初期サポート対象。CIで検証） |
| OS | Windows、Linux（CI/実機で検証後に正式確定） |
| デバイス | CPU（M0検証済み）、CUDA（未検証） |
| モデル | `google/embeddinggemma-2` のみ |
| モダリティ | テキスト、静止画像 |
| 出力次元 | 768固定 |
| dtype | FP32基準。FP32/BF16を選択可能。BF16の数値許容誤差は未確定 |
| FP16 | 禁止、要求時に明示エラー |
| 音声エンコーダー | 原則未ロード (`audio_config=None`) |
| デフォルト推論 | 推論専用、重みは学習済み |
| モデル配布 | 重みは同梱せずHugging Faceから取得 |

注：依存ライブラリの最低バージョンは動作確認によって確定し、未検証値を互換保証しない。

## 4. APIマトリックス

| API | 優先 | v0.1.0 | 仕様 / 制限 |
| --- | --- | --- | --- |
| `create_model` | P0 | 対応 | 単一モデル。FP32/BF16。CPU/CUDA |
| `create_model_and_transforms` | P0 | 対応 | `(model, preprocess_train, preprocess_val)`。Tensor入力前処理 |
| `get_tokenizer` | P0 | M3実装 | `LongTensor[B,L]`。既定はpromptなし。公式`prompt_name`を指定可能 |
| `encode_image` | P0 | M2実装済み | `[B,3,H,W]` → `[B,768]`。CPU FP32検証済み、BF16測定済み、CUDA未検証 |
| `encode_text` | P0 | M3実装 | 公式special tokenとright-padding規約を満たす`LongTensor[B,L]` → `[B,768]` |
| `.eval()`, `.to()`, `.cpu()`, `.cuda()` | P0 | 対応 | `nn.Module`の契約に準拠 |
| `forward` | P1 | 現時点で実装しない | v0.1.0のP0推論APIに含めない |
| `get_logits` | P1 | 現時点で実装しない | CLIPの学習済みlogit scaleとの混同を避ける |
| `create_model_from_pretrained` | P1 | 対応予定 | OpenCLIPと同じ返却数を維持 |
| `list_models` | P1 | 対応予定 | 自ライブラリの対応モデルのみ |
| `list_pretrained` | P2 | 未対応 | 単一モデルのため優先度低 |
| `get_model_config` | P2 | 未対応 | OpenCLIP内部設定とは非同型 |
| `forward_intermediates` | - | 非対応 | モデル内部が異なる |
| `load_checkpoint`（OpenCLIP形式） | - | 非対応 | 重みレイアウト非互換 |
| `open_clip_train` / `ClipLoss` | - | 非対応 | 推論限定 |
| `lock_image_tower`, `lock_text_tower` | - | 非対応 | 学習目的の操作 |
| 音声・動画埋め込み | - | 非対応 | 将来拡張 |

`EmbeddingGemmaCLIP`がTransformers backendとProcessorを保持する。
backendのみをPyTorch子moduleとして登録し、Processorは登録しない。
`eval()`、`to()`、`cpu()`、`cuda()`は`nn.Module`から継承する。
M2で`encode_image()`、M3で`get_tokenizer()`と`encode_text()`を実装する。`forward()`と`get_logits()`は現時点で実装しない。

## 5. 公開パッケージ・名前空間

Pythonインポート名：`openclip_gemma`。ユーザーは `import openclip_gemma as open_clip` として利用する。

### 5.1 `create_model`

推奨公開シグネチャ（実装時にOpenCLIPとの差異を固定する）：

```python
create_model(
    model_name: str = "embeddinggemma-2",
    pretrained: str | None = "google/embeddinggemma-2",
    precision: str = "fp32",
    device: str | torch.device = "cpu",
    cache_dir: str | None = None,
    **kwargs,
) -> torch.nn.Module
```

**規則**
- 許可するモデル名は`embeddinggemma-2`（正規化後）と、明示的に対応を検証したエイリアスのみ。
- `pretrained`は`google/embeddinggemma-2`のみ受け付ける。`None`は明示的に`ValueError`とし、未学習モデルや公式重みへの暗黙解決を行わない。
- `precision`: `fp32`と`bf16`を受け付ける。FP32を数値基準とし、BF16の数値許容誤差は別途確定する。`fp16`は明示エラー。
- `device`: `cpu`、`cuda`、`cuda:N`、`torch.device`。利用不能なデバイスは明示エラー。
- 未対応のOpenCLIP引数を `**kwargs`で無視することは禁止。`TypeError`または`NotImplementedError`を返す。
- ダウンロード失敗、アクセス不能、モデル不整合の原因を元例外から保持する。

### 5.2 `create_model_and_transforms`

```python
create_model_and_transforms(
    model_name: str = "embeddinggemma-2",
    pretrained: str | None = "google/embeddinggemma-2",
    precision: str = "fp32",
    device: str | torch.device = "cpu",
    cache_dir: str | None = None,
    **kwargs,
) -> tuple[torch.nn.Module, Callable, Callable]
```

- **戻り値**：`(model, preprocess_train, preprocess_val)` を保証。
- **P0画像入力**：`preprocess_val(PIL.Image.Image) -> torch.float32[3,H,W]`。`preprocess_val(img).unsqueeze(0)`は将来実装する`encode_image`の入力契約に適合する。
- `preprocess_train`は学習augmentationを行わず、`preprocess_val`と同じ決定的変換を返す。
- v0.1.0は学習サポートなし。`preprocess_train`は評価用と同一の決定的変換とする。OpenCLIP標準のaugmentationと互換だとは主張しない。
- 画像変換は固定したモデルrevisionの公式画像Processorメソッドを利用し、RGB変換、uint8画素でのリサイズ、1/255 rescaleの順序を保つ。返却TensorをProcessorに再投入する際はリサイズとrescaleを無効にし、二重適用を防ぐ。
- `H,W`は画像アスペクト比により異なる。固定サイズ化、crop、paddingは、公式経路との数値的一致を検証するまで行わない。
- 通常の`torch.stack()`による`[B,3,H,W]` batchは全画像の前処理後H/Wが同じ場合に限る。異なるH/Wの画像はdense Tensor batchにできず、可変寸法batchの専用APIは本P0契約に含めない。
- **P1将来拡張**：公式Processor feature mappingを直接受け取るAPIを検討する。P0のOpenCLIP互換Tensor入力とは別APIとする。
- **M0.1検証結果**：固定revision `914f7f89142e33e77833254d9c9b90c3cef7303b`で、公式メソッドを用いたTensor前処理後、Processorの`do_resize=False, do_rescale=False`で全feature mappingがPIL経路と完全一致した。RGB、グレースケール、wide、tall各画像のFP32埋め込みは最大絶対誤差`2.09e-7`〜`3.23e-7`、cosine similarity `1.0`。同一H/W batchも閾値内で一致。異なるH/Wを一つのdense Tensorにstackできるとは確認されていない。

### 5.3 `get_tokenizer`

```python
get_tokenizer(
    model_name: str = "embeddinggemma-2",
    *,
    prompt_name: str | None = None,
    context_length: int | None = None,
) -> EmbeddingGemmaTokenizer
```

- 入力：単一の`str`または空でない`Sequence[str]`。空文字列は受け付け、空バッチは`ValueError`。
- 出力：`torch.LongTensor[B,L]`。`L`はそのバッチで最長のtokenized sequence長で、右側だけをPADする。OpenCLIP tokenizerの固定context長paddingとは異なる。
- 特殊tokenは固定revisionの公式Tokenizerが付与する。確認値はPAD ID `0`、BOS ID `2`、EOS ID `1`、right padding。
- デフォルトではタスクpromptを付けない。`prompt_name`に固定revisionのSentence Transformers設定で定義された名前（例：`"SearchQuery"`、`"Classification"`、`"Document"`）を指定すると、定義済みprefixを**トークン化前に**一度だけ付ける。未定義のprompt名は拒否し、すでに公式prefixで始まる文字列にpromptを重ねることも拒否する。
- `context_length`は切り詰め上限であり、固定padding幅ではない。省略時はモデル設定の`text_config.max_position_embeddings=262144`を使う。TokenizerとSentence Transformersの`model_max_length`/`max_seq_length`は実用上の上限ではなく巨大な番兵値だったため採用しない。BOS/EOSを含み最低2 token必要。上限付近の長大系列による実行時メモリ・速度は未検証。
- truncate時は公式Tokenizerの挙動を使い、試験した`context_length=16`ではBOSを維持して末尾をEOSにした。
- **PAD IDとattention mask**：M0で試験した通常テキストでは`input_ids != 0`が公式maskと一致した。一方、公式Tokenizerは本文中の`<pad>`を有効token ID `0`として符号化する。この列はPADとpaddingをtoken IDだけで区別できないため、`get_tokenizer()`はこの入力を明示的に拒否する。`encode_text()`もPADはright-padding suffixにのみ許可し、BOS/EOSと範囲を検証する。
- 直接構築したTensorについて、PAD ID `0`を本文tokenとして使用することはできない。token ID-onlyのAPIでは末尾PADが本文かpaddingかを判別できないため、PADを専用のpadding IDとして予約することが入力契約である。
- Token IDをdecodeして再tokenizeすることは禁止。`encode_text()`は受け取ったIDを変更せず、復元したattention maskとともにbackendへ渡す。
- 例：`get_tokenizer("embeddinggemma-2", prompt_name="SearchQuery")`。promptなしは`get_tokenizer("embeddinggemma-2")`。

### 5.4 `model.encode_image`

```python
encode_image(image: torch.Tensor, normalize: bool = False) -> torch.Tensor
```

- 入力：`preprocess_val`で作成したfloat32 `[B,3,H,W]` Tensor。画素値は`[0,1]`、deviceはmodel backendと一致させる。
- `H,W`はimage patch sizeの倍数で、patch総数は公式Processorの上限以下であること。
- 出力：`[B,768]`、数値はfinite、デバイスはモデルと一致。
- `normalize=False`：公式Sentence Transformers pipelineのMean Pooling後の`Normalize`層を再現し、backend固有のネイティブ埋め込みを返す。未正規化vectorを人工的に復元しない。
- `normalize=True`：native embeddingにL2再正規化を適用。
- 無効なshape / dtype / deviceに対して明示例外。暗黙のCPU往復やPIL復元は禁止。
- 画像前処理は`create_model_and_transforms()`が返す公式処理と一致する決定的変換を利用する。P0のTensor経路で再びProcessorへ渡す場合は`do_resize=False, do_rescale=False`とする。
- Processorには各画像を個別groupにして渡し、`input_ids`、`attention_mask`、`pixel_values`、`image_position_ids`をbackend deviceへ移動する。attention maskを用いてtoken次元をmean-poolする。
- **M0.1検証結果**：公式処理順に従ったTensor前処理は、RGB・グレースケール・wide・tallの4画像で公式PIL経路とfeature単位で完全一致した。埋め込み最大絶対誤差は`2.09e-7`〜`3.23e-7`、cosine similarityは`1.0`。ただし、異なるH/Wを通常のdense Tensor batchにstackすることはできない。
- **M2検証結果**：固定revision `914f7f89142e33e77833254d9c9b90c3cef7303b`、CPU FP32でSentence Transformersの画像embeddingと比較。RGB、グレースケール、wide、tall、同一shape batch、`normalize=True`の最大絶対誤差は`2.09e-7`〜`3.23e-7`、cosine similarityは全て`1.0`（丸め前の浮動小数誤差を含む）。FP32暫定閾値を全て満たした。
- BF16では同じCPU/reference条件で最大絶対誤差`0.0`、cosine similarityは約`1.0`だった。FP32の閾値は適用せず、BF16の正式な許容誤差を確定するものではない。
- **M2未検証**：CUDA実行。画像Tensorのdeviceはmodelと一致させる。CPU/CUDA間の暗黙転送は行わない。

### 5.5 `model.encode_text`

```python
encode_text(text: torch.Tensor, normalize: bool = False) -> torch.Tensor
```

- 入力：トークンID `torch.LongTensor[B,L]`。
- 出力：`[B,768]`、finite、デバイスはモデルと一致。
- Tensorはモデルと同じdevice、モデル語彙の範囲内で、各行が公式BOSで始まり、公式EOSで終わり、PAD IDはright-padding suffixにのみ存在する必要がある。列長はモデルの`max_position_embeddings`以下。
- attention maskは`input_ids != pad_token_id`から構築する。任意の外部TensorのPAD ID本文は判別不能なため許可しない。タスクpromptは`encode_text()`では付与せず、必ずtokenizer生成時に適用する。Token IDはdecode・再tokenizeしない。
- backendへ`input_ids`とattention maskを渡し、公式Sentence Transformers経路と同じattention-mask付きMean Poolingおよびnative L2 Normalizeを行う。
- `normalize`規則は `encode_image` と同様。
- **M0検証結果**：promptなし、SearchQuery、Classificationの低レベルTransformers経路は、Mean Pooling・L2 Normalize後にSentence Transformers出力と最大絶対誤差`7.9768e-7`〜`1.3069e-6`、cosine similarity約`1.0`で一致した。
- **M3実測結果**：固定revision `914f7f89142e33e77833254d9c9b90c3cef7303b`、CPU FP32で英語・日本語・空文字列を含むbatch、promptなし/SearchQuery/Classification、`normalize=False/True`をSentence Transformersと比較した。最大絶対誤差は`1.27e-6`〜`3.99e-6`、cosine similarityは全行`>=0.9999998`で、暫定閾値を満たした。EOS special token単体も最大絶対誤差`1.20e-7`で一致した。
- 同revisionのCPU BF16では全比較の最大絶対誤差`0.0`、cosine similarityは約`1.0`だった。これは測定結果であり、BF16正式許容閾値を定めるものではない。
- CUDAは利用できず、統合テストはskipされた。長さ上限近くの長大系列についても実行速度・メモリの検証はしていない。

### 5.6 `forward` （P1・現時点で実装しない）

- 入力：`forward(image=None, text=None)`。
- どちらか一方のみ入力可能。両方`None`は`ValueError`。
- 出力形式：`(image_features | None, text_features | None, logit_scale)` を基本とするが、**logit_scaleはOpenCLIPと等価な学習済み値ではない**。
- `output_dict=True`に相当する返却はP1検証後に限定対応。
- v0.1.0のP0実装およびM0検証の対象外とする。互換を装うスケール定数を追加しない。

### 5.7 `get_logits`（P1・現時点で実装しない）

- EmbeddingGemma 2のL2正規化特徴の内積を返す。`[B_image,B_text]`。
- OpenCLIPの温度パラメータの再現はしない。
- v0.1.0のP0実装およびM0検証の対象外とする。追加のスケールを暗黙に導入しない。

### 5.8 PyTorch操作

- クラスは`torch.nn.Module`を継承し、バックエンドをsubmoduleとして登録。
- `.eval()`は評価モードへ移行。
- `.to()`, `.cpu()`, `.cuda()`はバックエンドを含め実際に重みを移動。
- `.half()`はFP16が非推奨かつ数値破綻の恐れがあるため、明示的に拒否する。`.to(dtype=torch.float16)`も拒否する。
- `.train()`はPyTorch APIとして存在するが、学習機能の互換保証はしない。必要なら警告を発する。

## 6. 前処理・タスク指示の基本原則

1. モデル固有の画像処理を参照モデルと一致させる。プロセッサーのリサイズ方式・画素正規化・画像トークン化の一致を確認する。
   - P0では公式処理と一致する決定的なTensor前処理を使用する。TensorへのProcessor再適用ではresize/rescaleを無効にする。
   - 前処理後のH/Wが画像ごとに異なることがある。異なる寸法をpadding/cropで揃える処理は数値的一致を検証するまで導入しない。
   - Processor feature mappingの直接入力はP1の拡張APIとし、P0 Tensor入力と混在させない。
2. テキスト指示プレフィックスはトークン化前に適用する。
3. マルチモーダル画像対テキストの最適プロンプトは、公式サンプルとベンチマークの比較で確定する。
4. 公式のタスクプレフィックス一覧（Classification、SearchQuery等）を目的と対応づける。無条件にすべての用途で1種類としない。
5. 768次元以外はv0.1.0のAPI契約外。将来MRLを追加する場合は切り詰め後に再正規化し、両モダリティの次元を一致させる。

M0ではpromptなし/SearchQuery/Classificationのいずれも、対応する公式Sentence Transformers出力とFP32暫定閾値内で一致した。promptの違いで埋め込みは変化するが、使用画像は合成テスト画像のため、既定promptの選定には使わない。

## 7. エラー契約

| 状況 | 例外 |
| --- | --- |
| 未知のモデル名 | `ValueError` |
| 未対応・危険なprecision（FP16） | `ValueError` |
| 未対応のfactoryオプション | `NotImplementedError`または`TypeError` |
| 画像入力がTensorでない | `TypeError` |
| 画像Tensorのshape / patch layout異常 | `ValueError` |
| 画像Tensorのdtypeがfloat32でない | `TypeError` |
| 画像TensorにNaN/Infまたは`[0,1]`範囲外値 | `ValueError` |
| 画像Tensorとbackendのdevice不一致 | `ValueError` |
| Token IDのdtype/shape異常 | `TypeError` / `ValueError` |
| Token ID範囲、BOS/EOS、padding構造異常 | `ValueError` |
| `get_tokenizer`への未知prompt、prompt二重適用、本文PAD token | `ValueError` |
| Tokenizerへの空batch・不正型入力 | `ValueError` / `TypeError` |
| 利用できないCUDA指定 | `RuntimeError` |
| モデル取得・ロード失敗 | `RuntimeError`（原因を`raise ... from ...`で保持） |
| 異なるH/Wの画像Tensorをdense batch化 | 通常の`torch.stack()`不可。padding/resizeによる自動補正は行わない |
| 学習・内部中間表現API | `NotImplementedError` |

シークレット、個人情報、ローカル絶対パスを例外文やログに不用意に残さない。

## 8. 受入テスト仕様

### T01. ファクトリ戻り値
`create_model_and_transforms`は3要素を返す。`model`は`nn.Module`、残り2要素はcallable。

### T02. OpenCLIP方式の画像入力
`preprocess_val(PIL.Image.open(...))`はfloat32の`[3,H,W]` Tensorを返す。M2では4種類の画像と同一shape batchを`encode_image`へ渡し、CPU FP32で公式出力との一致を検証する。

### T03. OpenCLIP方式のテキスト入力
`get_tokenizer(...)(["a cat", "a dog"])`は`LongTensor[2,L]`で、`encode_text`は`[2,768]`を返す。

### T04. バッチと順序
画像とテキストのバッチ順が保持され、単発推論とバッチ推論が各要素で許容差内で一致する。

### T05. 正規化
`normalize=True`で各行ノルムが`1 ± 1e-4`（FP32基準）。BF16は別に実測許容誤差を決める。

### T06. 参照モデル一致
同じ画像・テキスト・モデル重み・プロンプト・dtypeを使用し、埋め込みを比較。FP32初期閾値は最大絶対誤差`1e-4`、コサイン類似度`>=0.9999`を**暫定合格案**とし、実機検証で確定する。BF16は別の閾値を設定する。

### T07. 数値健全性
全てfinite、NaN/Infなし。異常入力・半精度指定は明示エラー。

### T08. PyTorchデバイス
`eval`, `to`, `cpu`, `cuda`が内部バックエンドにも反映される。CPU/CUDAそれぞれのテストを分離。

### T09. 典型的な互換コード
以下の一連の処理が追加の内部知識なしで動作する。

```python
import torch
from PIL import Image
import openclip_gemma as open_clip

model, _, preprocess = open_clip.create_model_and_transforms(
    "embeddinggemma-2", pretrained="google/embeddinggemma-2", device="cpu"
)
tokenizer = open_clip.get_tokenizer("embeddinggemma-2")
image = preprocess(Image.open("sample.jpg").convert("RGB")).unsqueeze(0)
text = tokenizer(["a cat", "a dog"])
with torch.inference_mode():
    im = model.encode_image(image, normalize=True)
    tx = model.encode_text(text, normalize=True)
    scores = im @ tx.T
assert scores.shape == (1, 2)
```

### T10. 軽量CIと実モデルCI
- 軽量CI：型・モック・エラー契約・ビルド・インストール、重みダウンロード不要。
- 実モデルCI：手動または定期実行、参照埋め込み比較、テキスト/画像実データ、CPU/CUDA（環境がある場合）。

## 9. 推奨ファイル構成

```text
openclip-gemma/
├── pyproject.toml
├── README.md
├── LICENSE
├── docs/
│   └── compatibility-spec-v0.1.0.md
├── src/openclip_gemma/
│   ├── __init__.py
│   ├── factory.py
│   ├── model.py
│   ├── tokenizer.py
│   ├── transforms.py
│   ├── exceptions.py
│   └── _backend.py
├── examples/
│   ├── zero_shot.py
│   └── compare_reference.py
└── tests/
    ├── test_factory.py
    ├── test_tokenizer.py
    ├── test_encode_image.py
    ├── test_encode_text.py
    ├── test_compatibility.py
    └── test_reference_integration.py
```

## 10. マイルストーンとリリースゲート

- **M0**：参照モデルを公式APIで稼働し、テキスト・画像の基準出力を収集。M0.1でTensor画像経路と制約を検証し、P0/P1画像API契約を決定済み。
- **M1**：ファクトリと画像前処理を実装。T01とT02の前処理部分を検証する。
- **M1.1**：`EmbeddingGemmaCLIP` wrapperにbackend/Processorを保持し、標準PyTorch操作を伝播する。固定revision統合テストでモデルロード・前処理一致を確認する。
- **M2**：`encode_image`のTensor検証、Processor feature mapping、Mean Pooling、native normalizationを実装。CPU FP32の公式一致とCPU BF16の測定を完了。CUDAは未検証。
- **M2**：`encode_image`の実装とCPU FP32/BF16検証、完了。
- **M3**：`get_tokenizer`、公式task prompt、Attention Mask復元、`encode_text`、参照数値比較、T03/T04/T06/T07/T08。
- **M4**：README・互換例・制限の整理、GitHub Actions、Wheel build、未検証環境の明示。
- **M4**：READMEに互換例・制限・数値スコアの意味、GitHub Actions、Wheel build。
- **Release v0.1.0**：P0 APIが全受入テストを満たす。P1未完了のものは明示的な非対応として公開。

## 11. 未決定事項

1. テキスト既定promptはなしとした。用途別promptを選ぶ最適方針はタスクベンチマークの対象外。
2. token ID-only APIではPAD IDを本文に使えない制約がある。PAD IDの本文利用とpaddingを両方厳密に扱うにはattention maskを別入力とする拡張が必要。
3. 対応するTransformersの最低バージョン。検証環境はTransformers 5.20.1。
4. CIで実際にサポートを保証するOS / CUDA組み合わせ。Python初期対象は3.12。CUDAは実行可能な環境で検証する。
5. P0 dense image Tensor batchで異なるH/Wを扱う方法は未解決。固定サイズ化・paddingは数値的一致の検証が必要であり、現時点では導入しない。

## 12. 情報源

- OpenCLIP factory: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/factory.py
- OpenCLIP model: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py
- OpenCLIP tokenizer: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/tokenizer.py
- OpenCLIP exports: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/__init__.py
- EmbeddingGemma 2 official model card: https://huggingface.co/google/embeddinggemma-2

**注**：本書はv0.1.0に対する**設計仕様案**であり、実装済み・数値検証済みという意味ではない。公開版仕様はM0〜M3の検証結果を反映して更新する。
