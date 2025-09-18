# Git/GitHub 作業マニュアル（2025ゼミ用）

## 0. 環境準備
- 可能なら作業ディレクトリは **OneDrive外** に置く  
- `.gitignore` を設定（例：`*.csv`, `__pycache__/`, `.ipynb_checkpoints/`）  
- 秘密情報（APIキーなど）は絶対にコミットしない

---

## 1. 作業開始前に main を最新化
```bash
git switch main
git pull origin main
```

---

## 2. 作業ブランチを作成
```bash
git switch -c feature/<名前>/<内容> main
```

例：
```bash
git switch -c feature/aze-aggdata-cleanup main
```

---

## 3. ファイルを編集 → 変更をコミット
```bash
git add -A
git commit -m "feat: 集計処理に欠損値補完を追加"
```

---

## 4. リモートにブランチを送る → PR作成
```bash
git push -u origin feature/aze-aggdata-cleanup
```
- GitHubで「Compare & pull request」ボタンを押す  
- PRタイトルと説明を記入（Before/After、影響範囲、確認方法を簡単に）

---

## 5. 最新 main を取り込む（PR中にmainが進んだら）

### rebase（履歴を綺麗に保つ方法）
```bash
git fetch origin
git rebase origin/main
# 競合を解決 → git add → git rebase --continue
git push -f
```

### merge（シンプル）
```bash
git fetch origin
git merge origin/main
git push
```

---

## 6. PRがマージされたら
```bash
git switch main
git pull origin main
git branch -d feature/aze-aggdata-cleanup
```

---

## 便利コマンド
```bash
git status             # 変更の確認
git branch -vv         # ブランチと追跡先の確認
git log --oneline --graph --decorate --all
```

---

## 命名規則（例）
- `feature/aze-...` → 新機能や分析追加
- `fix/aze-...`     → 不具合修正
- `docs/aze-...`    → ドキュメント修正
