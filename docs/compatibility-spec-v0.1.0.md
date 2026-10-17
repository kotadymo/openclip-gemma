# EmbeddingGemma OpenCLIP 互換性仕様書

- **文書ID**: EGOC-SPEC-001
- **対象リリース**: v0.1.0
- **状態**: Draft / M0・M0.1完了。M1.1 wrapper実装済み。画像・テキスト推論は未実装
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
| `get_tokenizer` | P0 | 条件付き | Tensor出力と順序・バッチ処理。モデル固有プレフィックス |
| `encode_image` | P0 | 対応 | `[B,3,H,W]` → `[B,768]` |
| `encode_text` | P0 | 対応 | `[B,L]` → `[B,768]` |
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

M1.1時点では`EmbeddingGemmaCLIP`がTransformers backendとProcessorを保持する。
backendのみをPyTorch子moduleとして登録し、Processorは登録しない。
`eval()`、`to()`、`cpu()`、`cuda()`は`nn.Module`から継承する。
`encode_image()`と`encode_text()`の公開シグネチャは定義済みだが、推論実装は後続マイルストーンまで`NotImplementedError`とする。

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
get_tokenizer(model_name: str = "embeddinggemma-2") -> Callable
```

- 入力：`str | list[str]`。空リストも対応し、空バッチTensorを返すか例外にするかを実装テストで確定（暫定案：明示エラー）。
- 出力：`torch.LongTensor[B,L]`、右パディングを基本とし、モデル固有のPAD/EOSに準拠。
- **M0検証結果**：検証したモデルrevisionではPAD IDは`0`、padding sideはright。タスクpromptなし/SearchQuery/Classificationの各サンプルで、`input_ids != pad_token_id`から得たmaskが公式`attention_mask`と一致した。これは試験したtokenizer生成列についての確認であり、任意の外部token ID列での一致は保証しない。
- タスクプレフィックスは**トークン化前に**適用。初期値は、画像とテキストの横断比較で使用する公式推奨の設定を検証した上で固定する。検証なしに`SearchQuery`や`Classification`を流用しない。
- トークンIDをデコードして再トークン化する変換は原則禁止。トークン化とモデル入力の低レベル橋渡しを実装する。
- プロンプト選択の変更が必要な場合、将来の拡張設定として明示的な`text_task`等を検討。v0.1.0ではトークナイザーインスタンスとモデルのプロンプト設定の整合性を最優先。
- 注意：トークンIDだけでは任意のカスタムプレフィックスとメタデータを保存できない。必要ならトークナイザー側で完全な入力列として符号化する。

### 5.4 `model.encode_image`

```python
encode_image(image: torch.Tensor, normalize: bool = False) -> torch.Tensor
```

- 入力：前処理済み`[B,3,H,W]`、浮動小数点Tensor。
- 出力：`[B,768]`、数値はfinite、デバイスはモデルと一致。
- `normalize=True`：L2再正規化（ゼロベクトルは明示的に検査）。
- `normalize=False`：モデルのネイティブ埋め込みを維持。参照モデルが正規化済みでも逆正規化しない。
- 無効なshape / dtype / deviceに対して明示例外。暗黙のCPU往復やPIL復元は禁止。
- 画像前処理は`create_model_and_transforms()`が返す公式処理と一致する決定的変換を利用する。P0のTensor経路で再びProcessorへ渡す場合は`do_resize=False, do_rescale=False`とする。
- **M0.1検証結果**：公式処理順に従ったTensor前処理は、RGB・グレースケール・wide・tallの4画像で公式PIL経路とfeature単位で完全一致した。埋め込み最大絶対誤差は`2.09e-7`〜`3.23e-7`、cosine similarityは`1.0`。ただし、異なるH/Wを通常のdense Tensor batchにstackすることはできない。
- **M1.1実装状態**：`EmbeddingGemmaCLIP.encode_image()`はシグネチャのみ実装し、推論は`NotImplementedError`。この段階ではT02の前処理一致を統合テストで確認し、埋め込み出力は主張しない。

### 5.5 `model.encode_text`

```python
encode_text(text: torch.Tensor, normalize: bool = False) -> torch.Tensor
```

- 入力：トークンID `torch.LongTensor[B,L]`。
- 出力：`[B,768]`、finite、デバイスはモデルと一致。
- 注意：OpenCLIPのTensor入力から、EmbeddingGemmaのattention mask、特殊トークン、タスクプロンプトの意味を再現する必要がある。
- `get_tokenizer()`以外から渡されたTensorも正当なID範囲であれば受け付ける。ただし「プレフィックス未付与」を推測して自動修正しない。
- `normalize`規則は `encode_image` と同様。
- **M1.1実装状態**：メソッドのシグネチャのみ定義し、推論は`NotImplementedError`。

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
| 画像Tensorのshape異常 | `ValueError` |
| Token IDのdtype/shape異常 | `TypeError` / `ValueError` |
| 利用できないCUDA指定 | `RuntimeError` |
| モデル取得・ロード失敗 | `RuntimeError`（原因を`raise ... from ...`で保持） |
| 異なるH/Wの画像Tensorをdense batch化 | `ValueError`（将来の`encode_image`で検証） |
| 学習・内部中間表現API | `NotImplementedError` |

シークレット、個人情報、ローカル絶対パスを例外文やログに不用意に残さない。

## 8. 受入テスト仕様

### T01. ファクトリ戻り値
`create_model_and_transforms`は3要素を返す。`model`は`nn.Module`、残り2要素はcallable。

### T02. OpenCLIP方式の画像入力
`preprocess_val(PIL.Image.open(...))`はfloat32の`[3,H,W]` Tensorを返す。M1では前処理を検証し、`encode_image`への接続と`[1,768]`出力は後続マイルストーンで検証する。

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
- **M1.1**：`EmbeddingGemmaCLIP` wrapperにbackend/Processorを保持し、標準PyTorch操作を伝播する。encode APIはシグネチャのみ定義し、`NotImplementedError`とする。固定revision統合テストでモデルロード・前処理一致を確認する。
- **M2**：Token ID経路とテキストプロンプトの実装、T03。
- **M3**：数値一致・バッチ・デバイス・失敗時挙動、T04–T08。
- **M4**：READMEに互換例・制限・数値スコアの意味、GitHub Actions、Wheel build。
- **Release v0.1.0**：P0 APIが全受入テストを満たす。P1未完了のものは明示的な非対応として公開。

## 11. 未決定事項

1. テキストと画像のCLIP用途に対する既定prompt。M0の合成画像比較では選定しない。
2. 任意の外部token ID Tensorからattention maskを復元できる範囲。M0では公式tokenizerが生成した例のみ一致を確認。
3. 対応するTransformersの最低バージョン。M0はTransformers 5.20.1で実行。
4. CIで実際にサポートを保証するOS / CUDA組み合わせ。Python初期対象は3.12。CUDAは未検証。
5. P0 dense Tensor batchで異なるH/Wを扱う方法は未解決。固定サイズ化・paddingは数値的一致の検証が必要であり、現時点では導入しない。

## 12. 情報源

- OpenCLIP factory: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/factory.py
- OpenCLIP model: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py
- OpenCLIP tokenizer: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/tokenizer.py
- OpenCLIP exports: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/__init__.py
- EmbeddingGemma 2 official model card: https://huggingface.co/google/embeddinggemma-2

**注**：本書はv0.1.0に対する**設計仕様案**であり、実装済み・数値検証済みという意味ではない。公開版仕様はM0〜M3の検証結果を反映して更新する。
