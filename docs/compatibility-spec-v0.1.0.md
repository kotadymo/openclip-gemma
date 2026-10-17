# EmbeddingGemma OpenCLIP 互換性仕様書

- **文書ID**: EGOC-SPEC-001
- **対象リリース**: v0.1.0
- **状態**: Draft / M0は部分検証済み。画像API契約は未決定
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
| dtype | FP32を基準とする。BF16は別途検証後に対応を確定 |
| FP16 | 禁止、要求時に明示エラー |
| 音声エンコーダー | 原則未ロード (`audio_config=None`) |
| デフォルト推論 | 推論専用、重みは学習済み |
| モデル配布 | 重みは同梱せずHugging Faceから取得 |

注：依存ライブラリの最低バージョンは動作確認によって確定し、未検証値を互換保証しない。

## 4. APIマトリックス

| API | 優先 | v0.1.0 | 仕様 / 制限 |
| --- | --- | --- | --- |
| `create_model` | P0 | 対応 | 主要引数のみ。単一モデル |
| `create_model_and_transforms` | P0 | 対応 | `(model, preprocess_train, preprocess_val)` |
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
- `precision`: 初期基準は`fp32`。`bf16`は独立した数値検証が完了するまでサポートを主張しない。`fp16`は明示エラー。
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
- `preprocess_val(PIL.Image.Image) -> torch.Tensor[3,H,W]`、`preprocess_val(img).unsqueeze(0)`は`encode_image`に適合。
- v0.1.0は学習サポートなし。`preprocess_train`は評価用と同一の決定的変換とする。OpenCLIP標準のaugmentationと互換だとは主張しない。
- 画像変換はモデル固有前処理と一貫するよう実装し、二重正規化・二重リサイズを防ぐ。
- `H,W`は参照プロセッサーの検証済み入力サイズに従い、仕様確定時に固定。任意解像度への強制変更は非対応。
- **M0検証結果**：検証したモデルrevisionでは公式`AutoProcessor`は`input_ids`、`attention_mask`、`pixel_values`、`image_position_ids`を返し、`pixel_values`は`[B,3,H,W]`ではない。PIL入力の全processor featureを`AutoModel`へ渡す経路は公式Sentence Transformers出力とFP32暫定閾値内で一致した。したがって、現在のTensorのみの契約は未検証であり、M1開始前に再検討する。

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
- 画像前処理の経路と参照実装を完全に同一とする必要があるため、Tensor直接入力部分は最重要の技術検証項目。
- **M0検証結果**：生の浮動小数Tensorをprocessorへ渡す経路はPIL経路と`pixel_values`が最大絶対誤差`0.0040472`異なり、出力埋め込みの最大絶対誤差は`0.00079144`、cosine similarityは`0.9999833`だった。暫定最大誤差閾値`1e-4`を満たさない。正確性を優先し、processor feature mappingをAPIで保持するか、Tensor経路をさらに検証するまでM1を開始しない。

### 5.5 `model.encode_text`

```python
encode_text(text: torch.Tensor, normalize: bool = False) -> torch.Tensor
```

- 入力：トークンID `torch.LongTensor[B,L]`。
- 出力：`[B,768]`、finite、デバイスはモデルと一致。
- 注意：OpenCLIPのTensor入力から、EmbeddingGemmaのattention mask、特殊トークン、タスクプロンプトの意味を再現する必要がある。
- `get_tokenizer()`以外から渡されたTensorも正当なID範囲であれば受け付ける。ただし「プレフィックス未付与」を推測して自動修正しない。
- `normalize`規則は `encode_image` と同様。

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
| 学習・内部中間表現API | `NotImplementedError` |

シークレット、個人情報、ローカル絶対パスを例外文やログに不用意に残さない。

## 8. 受入テスト仕様

### T01. ファクトリ戻り値
`create_model_and_transforms`は3要素を返す。`model`は`nn.Module`、残り2要素はcallable。

### T02. OpenCLIP方式の画像入力
`preprocess_val(PIL.Image.open(...)).unsqueeze(0)`が`encode_image`に渡せ、`[1,768]`を返す。

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

- **M0**：参照モデルを公式APIで稼働、テキストと画像の基準出力収集。**完了判定は画像API契約の決定後**。現時点で高レベル/低レベル一致は確認済みだが、OpenCLIP形式Tensorとの互換性は未解決。
- **M1**：ファクトリと画像前処理の実装、T01/T02。
- **M2**：Token ID経路とテキストプロンプトの実装、T03。
- **M3**：数値一致・バッチ・デバイス・失敗時挙動、T04–T08。
- **M4**：READMEに互換例・制限・数値スコアの意味、GitHub Actions、Wheel build。
- **Release v0.1.0**：P0 APIが全受入テストを満たす。P1未完了のものは明示的な非対応として公開。

## 11. 未決定事項（実装前に要確認）

1. Tensor-only `[B,3,H,W]`契約を維持するか、公式processor feature mappingを公開するか。PIL processor feature mappingは公式出力と一致したが、raw Tensor経路は暫定最大誤差閾値を超えた。
2. テキストと画像のCLIP用途に対する既定prompt。M0の合成画像比較では選定しない。
3. 任意の外部token ID Tensorからattention maskを復元できる範囲。M0では公式tokenizerが生成した例のみ一致を確認。
4. モデル読み込みの最低対応Transformers/Sentence Transformers版。M0はTransformers 5.20.1 / Sentence Transformers 6.1.0で実行。
5. CIで実際にサポートを保証するOS / CUDA組み合わせ。Python初期対象は3.12。CUDAは未検証。

## 12. 情報源

- OpenCLIP factory: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/factory.py
- OpenCLIP model: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py
- OpenCLIP tokenizer: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/tokenizer.py
- OpenCLIP exports: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/__init__.py
- EmbeddingGemma 2 official model card: https://huggingface.co/google/embeddinggemma-2

**注**：本書はv0.1.0に対する**設計仕様案**であり、実装済み・数値検証済みという意味ではない。公開版仕様はM0〜M3の検証結果を反映して更新する。
