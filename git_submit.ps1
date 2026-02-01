# 将当前文件夹提交到指定的 GitHub 仓库
# 使用前请修改下面 $repo 为你的仓库地址，例如 "username/repo-name"
$repo = "Talent-L/Image_enhanced"

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== 提交到 GitHub ===" -ForegroundColor Cyan
Write-Host "目标仓库: https://github.com/$repo" -ForegroundColor Yellow
if ($repo -match "你的用户") {
    Write-Host "请先编辑本脚本，将 `$repo 改为你的 GitHub 仓库地址（如 username/repo-name）" -ForegroundColor Red
    exit 1
}

# 若尚未初始化
if (-not (Test-Path .git)) {
    git init
    Write-Host "已初始化 Git 仓库" -ForegroundColor Green
}

# 若尚未添加远程
$remotes = git remote 2>$null
if (-not $remotes -or $remotes -notmatch "origin") {
    git remote add origin "https://github.com/$repo.git"
    Write-Host "已添加远程 origin" -ForegroundColor Green
}

git add .
$status = git status --short
if (-not $status) {
    Write-Host "没有需要提交的更改" -ForegroundColor Yellow
    exit 0
}

git commit -m "Initial commit: 项目初始提交"
git branch -M main

$choice = Read-Host "GitHub 上该仓库是否已有内容？(y/n，空则默认 n)"
if ($choice -eq "y" -or $choice -eq "Y") {
    git pull origin main --allow-unrelated-histories
}
git push -u origin main
Write-Host "推送完成" -ForegroundColor Green
