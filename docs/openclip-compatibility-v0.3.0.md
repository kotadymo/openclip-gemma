# openclip-gemma v0.3.0 OpenCLIP推論API互換性仕様書

| 項目 | 内容 |
| --- | --- |
| 文書ID | EGOC-SPEC-003 |
| 対象 | openclip-gemma v0.3.0 |
| 状態 | v0.3.0実装契約（本開発依頼で承認済み） |
| 主目的 | OpenCLIPの代表的な推論コードを、最小限の変更でEmbeddingGemma 2へ移行可能にする |
| 現行実装の基準 | `kotadymo/openclip-gemma`、`ff3a892beea68c97b5d7452702302881fe2a5783`（v0.2.0） |
| OpenCLIP参照基準 | `mlfoundations/open_clip`、`7067b350050dc409bfdfdbb3f0cc6c9a1651c000`（2026-09-30のコミット） |
| 参照モデル | `google/embeddinggemma-2`、固定revision `914f7f89142e33e77833254d9c9b90c3cef7303b` |

## 1. 基本方針と非目標

**基本方針**は、(1) EmbeddingGemma 2公式推論経路との数値的一致、(2) OpenCLIPの呼び出し・戻り値互換性、(3) エラーの明確さ、(4) 性能、の順に優先すること。互換性のために意味が違う数値を黙って偽装しない。

v0.3.0では以下をまとめて対応する。

- **Factory API**：`list_models()`、`list_pretrained()`、`create_model_from_pretrained()`。
- **Model API**：`forward()`、`get_logits()`、`output_dict=True`。学習済みスケールの不在は明示する。
- **移行・品質**：OpenCLIP代表スクリプトの互換テスト、v0.2.0回帰テスト、移行ガイド、CPU/CUDA実モデル検証。

**対象外**：OpenCLIP全モデルの読み込み、CLIPの重み・token IDの流用、学習・ファインチューニング、学習済み`logit_scale`の再現、FP16、音声、OpenCLIPの内部ネットワーク構造、全Factoryオプションの無条件受け入れ。v0.2.0の画像検索・動画検索機能は維持し、今回の主題にはしない。

利用者は `import openclip_gemma as open_clip` を使用する。`import open_clip` の名前空間を横取り・上書きしない。したがって「一行も変えず完全互換」とは主張しない。

## 2. v0.2.0からの互換性ルール

- 既存の`create_model()`、`create_model_and_transforms()`、`get_tokenizer()`、`encode_image()`、`encode_text()`の既定の結果、引数順序、モデル名、デバイス指定を維持する。
- `create_model()`の既存の既定`pretrained="google/embeddinggemma-2"`を維持する。`pretrained=None`は引き続きエラーとする。
- 既存のモデルrevision、公式画像前処理、Gemma tokenizer、公式タスクprompt、768次元正規化埋め込みを変更しない。
- 画像形状の可変性とテキストの動的right paddingを維持する。CLIPの固定77トークン、前処理画像の固定サイズ、学習用変換を装わない。
- `iter_image_embeddings()`、`build_or_update_image_index()`、`search_images()`、`build_or_update_video_index()`、`search_video()`を回帰テスト対象とする。保存インデックス形式は変更しない。
- 未サポートの引数・モデル・チェックポイントを黙って無視せず、理由が分かる例外にする。

## 3. API互換性マトリックス

| API | v0.2.0 | v0.3.0目標 | 備考 |
| --- | --- | --- | --- |
| `create_model` | 部分互換 | 拡張 | `output_dict`と`logit_scale_mode`を明示的に追加 |
| `create_model_and_transforms` | 部分互換 | 拡張 | 3要素返却を維持 |
| `get_tokenizer` | 部分互換 | 維持 | Gemma token ID・動的padding |
| `encode_image` / `encode_text` | 部分互換 | 維持 | 公式埋め込みと数値一致 |
| `list_models` | 未実装 | 追加 | モデル名の一覧 |
| `list_pretrained` | 未実装 | 追加 | `(モデル名, tag)`、`as_str` |
| `create_model_from_pretrained` | 未実装 | 追加 | モデルと前処理の返却 |
| `forward` | 未実装 | 条件付き互換 | `cosine`モード指定時に使用可能 |
| `get_logits` | 未実装 | 条件付き互換 | コサイン類似度の行列 |
| `output_dict=True` | 未実装 | 条件付き互換 | `forward`の辞書返却 |
| `get_model_config`等 | 未実装 | v0.3.0必須外 | 余力があれば別PR |
| 学習API / 他モデル / FP16 | 非対応 | 非対応 | 正直に明記 |

## 4. Factory APIの確定候補仕様

### 4.1 `list_models()`

```python
list_models() -> list[str]
# ["embeddinggemma-2"]
```

固定の対応モデル名のみ返す。モデル重みのダウンロード、GPU初期化、Hubへの通信を行わない。返却順序は安定させる。登録されていないCLIPモデル名は列挙しない。

### 4.2 `list_pretrained(as_str=False)`

```python
list_pretrained(as_str: bool = False) -> list[tuple[str, str]] | list[str]
# False: [("embeddinggemma-2", "official")]
# True:  ["embeddinggemma-2:official"]
```

`official`は本ライブラリが導入する**論理的なエイリアス**であり、Google公式が提供するOpenCLIPのpretrainedタグではない。内部対応先は`google/embeddinggemma-2`の固定revision。`as_str`は`bool`のみ許容。列挙だけではHub通信しない。OpenCLIP同様の要素形状を優先する。

### 4.3 `create_model_from_pretrained()`

```python
create_model_from_pretrained(
    model_name: str = "embeddinggemma-2",
    pretrained: str = "official",
    precision: str = "fp32",
    device: str | torch.device = "cpu",
    *,
    return_transform: bool = True,
    cache_dir: str | None = None,
    output_dict: bool = False,
    logit_scale_mode: str = "strict",
) -> tuple[EmbeddingGemmaCLIP, EmbeddingGemmaPreprocess] | EmbeddingGemmaCLIP
```

- `return_transform=True`なら`(model, preprocess_val)`、`False`なら`model`だけを返す。train transformは返さない。
- `pretrained="official"`および既存の`"google/embeddinggemma-2"`を許容する。既存Factoryに対しても`"official"`というエイリアスを追加するが、従来の既定値は変更しない。
- `None`、未知のタグ、未知のモデル名、`hf-hub:`/`local-dir:`による任意読み込みは拒否する。
- モデル読み込みは既存のFactory経路を再利用する。前処理の新たな実装や重みの二重ロードは行わない。
- 本家OpenCLIPにある`weights_only`、`force_image_size`などの未対応オプションは無視せず明示的に拒否する。安全に意味を保てる引数のみ将来追加する。
- `output_dict`、`logit_scale_mode`は下記と共通仕様。`create_model()`と`create_model_and_transforms()`にも**明示的なキーワード引数**として追加し、`**kwargs`で曖昧に処理しない。

## 5. Model APIの確定候補仕様

### 5.1 スケール係数の扱い

EmbeddingGemma 2にはOpenCLIPの**学習済み`logit_scale`と等価な値がない**。v0.3.0では次の2モードだけをサポートする。

| `logit_scale_mode` | 挙動 |
| --- | --- |
| `"strict"`（既定） | `forward()`と`get_logits()`は、スケール未定義を説明する`NotImplementedError`を送出する。`encode_image()`・`encode_text()`は従来どおり利用可能。 |
| `"cosine"`（明示指定） | `logit_scale=1.0`の非学習済み係数を使う。`forward()`と`get_logits()`を使用可能にする。返すスコアは**未較正のコサイン類似度**であってOpenCLIPの学習済みlogitsではない。 |

- `1.0`は**明示的な出力スケール**であり、学習可能な`nn.Parameter`、CLIP互換のチェックポイント重み、または確率ではない。
- `logit_scale`という名称はOpenCLIPの`forward()`の戻り値に合わせる。返す値は**対数スケールではなく乗算係数そのもの**のscalar Tensor。GPU利用時はモデルと同じdeviceのfloat32 Tensor。`model.logit_scale.exp()`のようなCLIP固有アクセスの互換は保証しない。
- モードはモデルのFactory生成時に設定する。**strictからcosineへの暗黙フォールバックは禁止**。無効なモード値はモデル生成前に拒否する。
- v0.2.0の画像検索が使用するコサイン計算の意味・出力は変えない。

### 5.2 `forward(image=None, text=None)`

```python
forward(
    image: torch.Tensor | None = None,
    text: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor] | dict[str, object]
```

`logit_scale_mode="cosine"`が必要。`output_dict=False`の場合は`(image_features, text_features, logit_scale)`の**3要素タプル**を返す。`output_dict=True`の場合は次のキーを返す。

```python
{
    "image_features": image_features,  # [N, 768] or None
    "text_features": text_features,    # [M, 768] or None
    "logit_scale": scale,               # scalar Tensor, value 1.0
}
```

画像のみならテキスト特徴量は`None`、テキストのみなら画像特徴量は`None`。両方`None`なら`ValueError`。画像とテキストのバッチ数は一致不要。内部で既存の`encode_image(..., normalize=True)`、`encode_text(..., normalize=True)`を呼ぶ。**画像とテキストの特徴量同士の結合や暗黙のsoftmaxは行わない**。追加の`logit_bias`は返さない。

### 5.3 `get_logits(image, text)`

`logit_scale_mode="cosine"`でのみ利用可能。入力は両モダリティ必須。`N`枚の画像と`M`件のテキストに対し、次を返す。

```python
image_logits = image_features @ text_features.T  # [N, M]
text_logits = image_logits.T                      # [M, N]
return image_logits, text_logits
```

画像・テキストは既存エンコーダーで正規化する。スコアはコサイン類似度（確率ではない）。device・dtype・finite・次元を検証し、失敗時に黙ってCPUへ移動しない。`get_logits`の返却形式は`output_dict=True`に左右されず常に2要素タプル。

### 5.4 `output_dict`の入力場所

`output_dict: bool = False`を3つのFactoryに明示的に持たせ、生成された`model.output_dict`の値に基づいて**forwardだけ**の戻り値を切り替える。`forward(..., output_dict=True)`のような呼び出し単位の指定はv0.3.0の対象外。`get_logits()`の戻り値も変えない。bool以外は`TypeError`。

## 6. 非互換・エラー仕様

| 状況 | 期待動作 |
| --- | --- |
| `pretrained=None`／未知のタグ・モデル | `ValueError` |
| 未知のFactory引数 | `TypeError`（黙って無視しない） |
| `output_dict`がbool以外 | `TypeError` |
| 未知の`logit_scale_mode` | `ValueError` |
| strictモードで`forward`／`get_logits`を使用 | 説明付き`NotImplementedError` |
| cosineモードで両入力とも`None`の`forward` | `ValueError` |
| 不正な画像Tensor／token ID／device | 既存の明示的エラー規約を継承 |
| `precision="fp16"`／`.half()` | 引き続き拒否 |
| 任意のOpenCLIPチェックポイント・学習API | 非対応を明示 |

ただし不正な入力が複数重なった場合のエラー優先順位は、(1) Factory引数、(2) `forward`の両入力欠如、(3) strictモード、(4) 入力Tensor検証、の順とする。テストでは特にstrictモードが意図せずモデル推論を開始しないことを確認する。

## 7. 利用者向けサンプル（実装完了後に動作する目標）

```python
import torch
import openclip_gemma as open_clip

assert open_clip.list_models() == ["embeddinggemma-2"]
assert ("embeddinggemma-2", "official") in open_clip.list_pretrained()

model, preprocess = open_clip.create_model_from_pretrained(
    "embeddinggemma-2",
    pretrained="official",
    device="cpu",
    output_dict=True,
    logit_scale_mode="cosine",  # コサイン類似度であることを明示
)
model.eval()
tokenizer = open_clip.get_tokenizer("embeddinggemma-2")

# image = PIL.Image.Image
# image_tensor = preprocess(image).unsqueeze(0)
# text_tokens = tokenizer(["a mountain", "a cat"])
# with torch.inference_mode():
#     output = model(image_tensor, text_tokens)
#     image_logits, text_logits = model.get_logits(image_tensor, text_tokens)
# print(output["image_features"].shape)  # [1, 768]
# print(image_logits.shape, text_logits.shape)  # [1, 2], [2, 1]
```

使用例では`logit_scale_mode="cosine"`を必ず明示し、CLIP学習済みlogitsや分類確率と誤解させない。実行用の完全スクリプトを`examples/openclip_compatibility_v030.py`に用意し、外部画像を必要としない合成画像を既定入力とする。`--image`で利用者の画像も指定できる。

## 8. テスト設計と合否条件

### 8.1 単体テスト（モデル重みのダウンロード不要）

- Factory列挙APIの型・順序・`as_str`・未知入力・オフライン性。
- モデル生成の戻り値（`return_transform=True/False`）、エイリアス、旧引数互換、誤引数の拒否。
- mock backendで`forward`の両方／片方／入力なし、tuple／dict、scale scalarのdtype/device。
- `get_logits`のshape、転置一致、dot-product一致、別バッチサイズ、non-finite拒否。
- strictモードで**無推論・明示的失敗**すること、cosineモードの明示性。
- `__all__`、wheel内公開API、`import openclip_gemma as open_clip`の移行コード。

### 8.2 代表的なOpenCLIPコード10ケース

1. `list_models()`
2. `list_pretrained()`と`as_str=True`
3. `create_model_from_pretrained()`（前処理付き）
4. `return_transform=False`（モデルのみ）
5. `create_model_and_transforms()`と`encode_image()`
6. `get_tokenizer()`と`encode_text()`
7. 画像・文章の類似度行列
8. `model(image, text)`の3要素タプル
9. `output_dict=True`の辞書・単一モダリティ
10. `get_logits()`の画像→テキスト／転置の2行列

各ケースについて`A: import文とモデル名だけ変更`、`B: モデル固有の追加引数・前処理変更で動作`、`C: 意味の一致が不可能で非対応`のいずれかを明記。A/B/Cを強引に全Aとせず、差分行数・追加設定も記録する。比較するOpenCLIP元コードを固定し、実行可否は実測のみ記録する。

### 8.3 実モデル統合テスト

- CPU FP32の新API埋め込みが、同じ固定revisionの公式Sentence Transformers経路と一致すること。参考基準：最大絶対誤差`<=1e-4`、cosine `>=0.9999`（既存基準）。
- CPU BF16およびRTX 4070 Ti SUPER CUDA FP32/BF16を実機検証し、環境・最大誤差・dtype・device・メモリを記録。BF16はFP32閾値を機械的に流用せず、実測に合わせて別途許容差を文書化する。
- 新しい`forward()`の特徴量と既存`encode_*`の出力が一致すること。`get_logits()`と直接の正規化済み内積が一致すること。
- Windows／Ubuntu × Python 3.12／3.13／3.14（ただし3.14.1除外）のCIを通す。通常CIは実モデルダウンロードを要求しない。
- v0.2.0のバッチ埋め込み、画像検索、動画検索、インデックス互換性の回帰試験を通す。video extraの有無を分離テストする。

## 9. 実装順序

| 順序 | PR案 | 成果物 |
| --- | --- | --- |
| 1 | 仕様確定 | 本仕様書・本家参照SHA・期待テストケース |
| 2 | Factory API | `list_models`、`list_pretrained`、`create_model_from_pretrained`、テスト |
| 3 | Model API | `forward`、`get_logits`、`output_dict`、strict/cosineテスト |
| 4 | 互換性と回帰検証 | 10例の分類・実モデル検証・v0.2.0回帰試験 |
| 5 | ドキュメント | README・移行ガイド・数値と制限の明示 |
| 6 | リリース準備 | v0.3.0メタデータ、TestPyPI RC／本番workflow、wheel/sdist確認 |

本開発依頼では全工程を専用ブランチ`feature/v0.3.0-openclip-compatibility`上で進め、main向けDraft PRを1件作成する。PRはレビュー待ちとし、マージ・タグ作成・TestPyPI／PyPI公開・GitHub Release作成は行わない。バージョン更新はAPI・テスト・ドキュメントの準備後に行う。仕様変更時は変更理由と移行への影響を進捗文書とPRに記録する。

## 10. v0.3.0リリース完了条件

- [ ] 全必須Factory／Model APIと既存の公開APIが仕様どおり動く。
- [ ] `logit_scale_mode`のstrict／cosineの意味とエラーをテスト済み。
- [ ] OpenCLIP代表10ケースのA/B/C分類、差分行数、結果を公開済み。
- [ ] 固定EmbeddingGemmaモデルに対するCPU/CUDA実モデル検証済み。
- [ ] v0.2.0の画像・動画・バッチ機能に回帰なし。
- [ ] CI、Ruff、依存lock、wheel/sdist、METADATA/README整合性チェックが通過。
- [ ] 公開READMEに「まだ未公開」「公開後に更新」等の時点依存記述なし。
- [ ] 新規TestPyPIのRC版をクリーン環境で実インストール・動作確認済み。
- [ ] タグ対象main SHAでCIが成功済み。CI実行とタグpushの競合を避ける。
- [ ] PyPIのv0.3.0用Trusted Publisher・保護Environmentを確認済み。
- [ ] 正式PyPI公開後のwheel/sdist、パッケージ説明、GitHub Releaseを確認済み。

## 11. 参照URL

- openclip-gemma v0.2.0: https://github.com/kotadymo/openclip-gemma/tree/v0.2.0
- v0.2.0 Factory: https://github.com/kotadymo/openclip-gemma/blob/v0.2.0/src/openclip_gemma/factory.py
- v0.2.0 Model: https://github.com/kotadymo/openclip-gemma/blob/v0.2.0/src/openclip_gemma/model.py
- 既存互換性レポート: https://github.com/kotadymo/openclip-gemma/blob/v0.2.0/docs/openclip-compatibility-report.md
- OpenCLIP固定参照: https://github.com/mlfoundations/open_clip/tree/7067b350050dc409bfdfdbb3f0cc6c9a1651c000
- OpenCLIP factory: https://github.com/mlfoundations/open_clip/blob/7067b350050dc409bfdfdbb3f0cc6c9a1651c000/src/open_clip/factory.py
- OpenCLIP pretrained: https://github.com/mlfoundations/open_clip/blob/7067b350050dc409bfdfdbb3f0cc6c9a1651c000/src/open_clip/pretrained.py
- OpenCLIP model: https://github.com/mlfoundations/open_clip/blob/7067b350050dc409bfdfdbb3f0cc6c9a1651c000/src/open_clip/model.py

---

**承認済み決定**：本開発依頼により、`official`タグ名、`strict`をデフォルトとする方針、`forward()`と`get_logits()`を`logit_scale_mode="cosine"`指定時のみ使用可能とする方針が承認された。これらは本仕様の確定条件として実装・テスト・移行ガイドに反映する。
