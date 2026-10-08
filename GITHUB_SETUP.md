# 在 VS Code 中发布到 GitHub

本目录已有本地 Git 仓库，当前分支是 `main`。无需再次初始化。
新增的 `.gitignore` 会排除缓存、虚拟环境、生成的数据和实验结果；这些文件仍保留在本机。

## 上传哪些文件

保留 Python 源码、`tests/`、`island_toy` 中的源码和测试、Markdown 文档、
配置 JSON，以及根目录的 `cases.csv`。

忽略根目录的 `index/`、`grids/`、`runs/`、`dataset/`、`training/`、
`evaluation/`、`logs/`、`diagnostics/`，以及 `island_toy/` 中相应的
生成目录和 `terrains/`。同时忽略 Python 缓存、虚拟环境、本地编辑器设置和常见凭据文件。

没有统一忽略所有 CSV、JSON 或 PNG 文件，以便后续保存配置、样例和文档图片。
原始 ERA5/GEBCO 数据和 SWAN 程序位于本目录之外，本仓库不会包含它们。
换电脑后，需要按 `README.md` 准备外部数据、程序和模型依赖，并重新生成数据目录。
配置中含有本机路径，新电脑上需要调整。

## 首次提交与发布

1. 在 VS Code 中选择“文件 → 打开文件夹”，打开 `F:\TXG\CODE_WNE\TOY\v2`。
2. 按 `Ctrl+Shift+G` 打开“源代码管理”。变更列表应该只包含源码、文档和配置，
   不应出现生成的数据、模型权重或 `__pycache__`。
3. 在 VS Code 的终端中设置提交者信息。以下值必须替换成你自己的信息；
   不带 `--global`，只影响本项目。邮箱可以使用 GitHub 设置 → Emails 中提供的 noreply 邮箱。

   ```powershell
   git config user.name "你的名字或 GitHub 用户名"
   git config user.email "你的提交邮箱"
   ```

4. 在源代码管理中选择“暂存所有更改”（Changes 旁边的 `+`），
   输入提交说明 `Initial commit`，点击“提交”。提交会保存本地版本记录。
5. 按 `Ctrl+Shift+P`，搜索并执行 `Publish to GitHub`（发布到 GitHub）。
   按提示登录 GitHub，输入仓库名，例如 `swan-toy-v2`，选择私有或公开仓库。
   私有仓库适合先用于个人备份；公开仓库的文件任何人都能访问。
6. 发布成功后点击“在 GitHub 上打开”，检查源码已上传，生成目录未上传。

VS Code 的发布命令会创建 GitHub 仓库、配置远程连接并推送本地提交，
无需提前在 GitHub 网页上创建另一个仓库。

## 后续更新

修改文件后，打开源代码管理 → 检查变更 → 暂存 → 填写说明并提交 → 推送或同步更改。
提交保存本地记录，推送才会把记录上传到 GitHub。

## 检查忽略规则

在本目录的终端中执行：

```powershell
git status --short
git status --short --ignored
git check-ignore -v island_toy/training/1km/best.pt
```

第二条命令中以 `!!` 开头的项目就是已忽略的文件或目录。
若要忽略新的生成目录，给 `.gitignore` 添加相应路径即可；
例如 `/scratch/` 只忽略项目根目录的 `scratch` 文件夹。

`.gitignore` 对已经提交或暂存的文件不会自动生效。将来若误跟踪了生成目录，
需要先用 `git rm -r --cached 目录路径` 取消跟踪，再提交；该操作保留本机文件。
目前项目还没有首次提交，不需要做这一步。

官方说明：[VS Code：发布仓库到 GitHub](https://code.visualstudio.com/docs/sourcecontrol/repos-remotes)。
