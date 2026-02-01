# 将本文件夹提交到 GitHub 的步骤

请将下面命令中的 **`你的用户名/你的仓库名`** 替换为你实际的 GitHub 仓库地址（例如 `zhangsan/my-image-project`）。

---

## 方法一：在终端中逐条执行（推荐）

在 PowerShell 或 CMD 中，先进入项目目录：

```powershell
cd "e:\Postgraduate\Academic_research\Image_enhanced\Test_code"
```

然后依次执行：

### 1. 初始化 Git 仓库

```powershell
git init
```

### 2. 添加远程仓库（请替换为你的 GitHub 仓库地址）

```powershell
git remote add origin https://github.com/你的用户名/你的仓库名.git
```

如果使用 SSH：

```powershell
git remote add origin git@github.com:你的用户名/你的仓库名.git
```

### 3. 添加所有文件并提交

```powershell
git add .
git commit -m "Initial commit: 项目初始提交"
```

### 4. 设置主分支并推送（若仓库已有内容可先拉取）

若 GitHub 上仓库是**空的**：

```powershell
git branch -M main
git push -u origin main
```

若 GitHub 上**已有文件**（如 README），先拉取再推送：

```powershell
git branch -M main
git pull origin main --allow-unrelated-histories
git push -u origin main
```

---

## 方法二：一键脚本（需先改仓库地址）

1. 用记事本或 VS Code 打开项目里的 `git_submit.ps1`。
2. 把其中的 `你的用户名/你的仓库名` 改成你的真实 GitHub 仓库地址。
3. 在 PowerShell 中执行：

```powershell
cd "e:\Postgraduate\Academic_research\Image_enhanced\Test_code"
.\git_submit.ps1
```

按提示选择「仓库是空的」或「仓库已有内容」。

---

## 注意事项

- **首次推送**可能需要登录 GitHub（浏览器弹窗或凭据管理器）。
- 若 **data** 目录下 `.h5` 或图片很大，建议使用项目中的 `.gitignore` 排除后再 `git add .`，否则推送可能很慢或失败。
- 若执行 `git push` 时报错 **Permission denied** 或 **Authentication failed**，请检查：
  - 使用 HTTPS 时：GitHub 账号与密码（或 Personal Access Token）。
  - 使用 SSH 时：本机是否已配置 SSH 公钥并添加到 GitHub。

完成以上步骤后，当前文件夹中的文件就会出现在你指定的 GitHub 仓库中。
