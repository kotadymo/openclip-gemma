# M6 TestPyPI 試験公開準備

## 状態

- **公開状況**：準備のみ。TestPyPIおよび本番PyPIへのアップロードは行っていない。
- **試験版**：`0.1.0rc1`（PEP 440 prerelease）
- **ビルド成果物**：
  - `dist/m6-testpypi/openclip_gemma-0.1.0rc1.tar.gz`
  - `dist/m6-testpypi/openclip_gemma-0.1.0rc1-py3-none-any.whl`
- **安定版設定**：`pyproject.toml`は`0.1.0`のまま。rc1の成果物を作る間だけ一時的に`0.1.0rc1`へ変更し、ビルド後に`0.1.0`へ戻した。
- **TestPyPIの名前確認**：`https://test.pypi.org/pypi/openclip-gemma/json`は2026-10-17にHTTP 404。アップロード直前に再確認する。

## main・CI・APIの確認

- 確認したmain SHAとローカルHEAD：`17fd6fd1270738a1e82d1e4a56e739c9162c4506`
- GitHub Actionsの最新成功run：[#39014004446](https://github.com/kotadymo/openclip-gemma/actions/runs/39014004446)、同じSHA、2026-10-17 07:32:19 UTC。
- `pyproject.toml`のauthorsは`{ name = "kotadymo" }`で、emailは指定されていない。
- wheel metadataの`Author`は`kotadymo`、`Author-email`はなし。
- 公開APIは`EmbeddingGemmaCLIP`、`create_model()`、`create_model_and_transforms()`、`get_tokenizer()`。READMEは現行APIと互換性制約を説明している。
- LICENSEはApache-2.0全文。固定revisionのモデルカードは`apache-2.0`を宣言し、OpenCLIPはMIT。wheel/sdistにモデル重みとOpenCLIP実装コードは含まれない。
- 既存CIは品質検査・ビルド・wheel検査用。TestPyPI公開workflowは別ファイルに作成済みだが、未commit・未設定・未実行。

## ビルドと検査

`uv build --out-dir dist/m6-testpypi`でsdistとwheelを生成した。wheel metadataには次が含まれることを確認した。

- `Version: 0.1.0rc1`
- `Author: kotadymo`
- `License-Expression: Apache-2.0`
- `Author-email`なし
- runtime依存はPillow、PyTorch、Transformers、torchvision。`test`、`integration`、`benchmark`はoptional extra。

成果物のSHA-256:

| ファイル | SHA-256 |
|---|---|
| `openclip_gemma-0.1.0rc1.tar.gz` | `BECC7AD2EC89C5AE70D9C0EA8FFEC9018AE7F61119452C79497715F065935090` |
| `openclip_gemma-0.1.0rc1-py3-none-any.whl` | `BF426BD319042DF2F6B6B9BE7B89CDF5416F9C2195365A7A1CFE645F3073F834` |

wheel/sdistの内容を直接検査し、License、README、metadata、Python packageだけが含まれること、モデル重みやOpenCLIP実装コードがないことを確認した。追加検査で`.dist-info/licenses/LICENSE`とsdist内`LICENSE`が公式全文と一致することも確認した。

依存関係は本番PyPIだけをindexに指定して、Python 3.12向けにdry-run解決した。37パッケージが解決され、インストールは行っていない。

```powershell
uv pip install --dry-run --system --python-version 3.12 `
  --default-index https://pypi.org/simple `
  "pillow>=13.0.0,<14" `
  "torch>=2.14.1,<3" `
  "transformers>=5.20.1,<6" `
  "torchvision>=0.29.1,<1"
```

## ユーザー承認後のアップロード方法

本書のコマンドは手順例であり、実行していない。ユーザーの明示的な承認後にだけアップロードする。

### 推奨：Trusted Publishing（OIDC）

継続的・自動化された公開にはTrusted Publishingを推奨する。GitHub Actionsから短命の認証情報を使い、長期API tokenを保管せずに済む。TestPyPI側にも本番PyPIとは別にTrusted Publisherを設定する必要がある。専用workflowは`.github/workflows/publish-testpypi.yml`に作成したが、TestPyPI publisherとGitHub environmentは未設定であり、workflowは未commit・未実行。

### TestPyPI Trusted Publisherの設定値

TestPyPIで初回公開前のpending publisherを登録する。プロジェクト情報URLが404であることを確認済みのため、account-level Publishing設定からproject作成用publisherを登録する。

| TestPyPI/GitHub設定 | 値 |
|---|---|
| Package/project name | `openclip-gemma` |
| GitHub owner | `kotadymo` |
| Repository | `openclip-gemma` |
| Workflow filename | `publish-testpypi.yml`（repository path: `.github/workflows/publish-testpypi.yml`） |
| GitHub environment | `testpypi` |

GitHub Web UIではRepository **Settings → Environments → New environment**から`testpypi`を作成し、required reviewersを設定する。TestPyPI側では[Account settings → Publishing](https://test.pypi.org/manage/account/publishing/)から上記owner/repository/workflow/environmentを登録する。projectがまだ存在しない場合はpending publisherとして登録する。publisher登録後も、project名は初回OIDC uploadまで予約されない。

現workflowの公開jobだけに`id-token: write`を付与し、ビルドjobは`actions: read`と`contents: read`のみ。公開jobはTestPyPIの`https://test.pypi.org/legacy/`へ配布物を送る。PyPI本番へのworkflowやtokenは含まれない。

workflowとpublisher設定がdefault branchに反映された後、GitHub Web UIの**Actions → Publish 0.1.0rc1 to TestPyPI → Run workflow**でrefに`v0.1.0rc1`を選び、確認欄で`publish-to-testpypi`を選択して実行する。事前にタグcommitのCI成功を待つ。tag pushでもworkflowは起動するが、`testpypi` environmentのrequired reviewer承認が公開jobの実行前に必要となる。

参考：[PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/)、[既存projectへのpublisher追加](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)、[OIDCによるproject作成](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)。

### 一回限りの手動試験：TestPyPI API tokenとTwine

Trusted Publishingを利用できない場合の代替手段。新しいworkflowはAPI tokenを使用しない。TestPyPI専用tokenを使う場合は可能な限り最小scopeにし、コード、`pyproject.toml`、`.pypirc`、shell historyに保存せず、Twineの対話promptへ入力する。

```powershell
uvx --from twine twine upload --repository testpypi `
  --username __token__ `
  dist\m6-testpypi\openclip_gemma-0.1.0rc1.tar.gz `
  dist\m6-testpypi\openclip_gemma-0.1.0rc1-py3-none-any.whl
```

参考：[Python Packaging User Guide — Using TestPyPI](https://packaging.python.org/en/latest/guides/using-testpypi/)。

## TestPyPIからのクリーン環境インストール・推論確認

TestPyPIにrc1をアップロードした後、新しいPowerShellで以下を実行する。依存は本番PyPIから先に導入し、TestPyPIからはパッケージ本体だけを`--no-deps`で導入するため、TestPyPIを依存解決sourceとして使わない。

```powershell
$venv = Join-Path $env:TEMP "openclip-gemma-testpypi-0.1.0rc1"
py -3.12 -m venv $venv
$python = Join-Path $venv "Scripts\python.exe"

& $python -m pip install --upgrade pip
& $python -m pip install --index-url https://pypi.org/simple `
  pillow torch transformers torchvision
& $python -m pip install --index-url https://test.pypi.org/simple/ `
  --no-deps "openclip-gemma==0.1.0rc1"
& $python -m pip check
```

importとCPU推論を検証する。

```powershell
@'
import importlib.metadata

import torch
from PIL import Image

import openclip_gemma

assert importlib.metadata.version("openclip-gemma") == "0.1.0rc1"
model, _, preprocess = openclip_gemma.create_model_and_transforms(device="cpu")
tokenizer = openclip_gemma.get_tokenizer()
image = Image.new("RGB", (256, 192), color=(80, 120, 180))
tokens = tokenizer(["青い背景にある赤い円"])

model.eval()
with torch.inference_mode():
    image_features = model.encode_image(preprocess(image).unsqueeze(0))
    text_features = model.encode_text(tokens)

assert image_features.shape == (1, 768) and torch.isfinite(image_features).all()
assert text_features.shape == (1, 768) and torch.isfinite(text_features).all()
print("CPU image/text inference passed", image_features.shape, text_features.shape)
'@ | & $python -
```

初回推論は固定revisionのHugging Faceモデル重みを取得するため、ネットワーク接続と十分なディスク容量が必要。モデル重みをwheel/sdistから取得するわけではない。

## version管理方針とリリース時の注意

- TestPyPI上の`0.1.0rc1`は本番PyPIの`0.1.0`を予約・消費しない。indexは別々。
- TestPyPIの同一versionは再アップロードせず、修正が必要なら`0.1.0rc2`へ進める。
- rc成果物は本番PyPIにアップロードしない。安定版は`pyproject.toml`を`0.1.0`にしたクリーンなrelease commitから再ビルドし、成果物を再検査する。
- 現在のmainは`pyproject.toml`と`uv.lock`がともに`0.1.0`のため、現在のHEADへrcタグを付けてもworkflowは拒否する。公開前にmain上へ`pyproject.toml`を`0.1.0rc1`にした候補commitを作り、`uv lock`でlockfileも更新する。CIの配布物検査とwheelインストール検査は`pyproject.toml`のversionを参照するためrc版を検証できる。CI成功後、そのcommitに`v0.1.0rc1`タグを付ける。試験後はstable用の`0.1.0`へ戻すcommitを別途作る。
- Workflowは`v0.1.0rc1`タグまたはタグを選択した明示的workflow_dispatchだけで起動する。タグcommitがmainの祖先であり、同一SHAに既存CIの成功runがある場合だけ先へ進む。pushタグとCIが競合してCIがまだ完了していなければ、workflowは安全側に失敗するため、CI成功後にタグrefから手動再実行する。
- TestPyPIでの検証後にREADME/version/changelogを確認し、TestPyPIと本番PyPIそれぞれのpublisher/token設定・project名を取り違えない。
- M6準備後の変更は未commitであり、記録したCI成功SHAはこのローカル文書・成果物・workflowを含まない。commit/push/tag作成および公開は実施していない。
