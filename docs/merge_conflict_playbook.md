# Merge conflict playbook

If your PR says "This branch has conflicts", follow this flow.

## 1) Create a safe branch for conflict resolution

```bash
git checkout -b fix/merge-conflicts
```

## 2) Bring in target branch changes

Example when your PR target is `main`:

```bash
git fetch origin
git merge origin/main
```

> If your project prefers rebasing, use `git rebase origin/main` instead.

## 3) List conflicted files

```bash
git status
```

Look for lines under **Unmerged paths**.

## 4) Open each conflicted file and resolve markers

Search markers quickly:

```bash
make conflict-check
```

You must remove all of these markers in every conflicted file:

- `<<<<<<< HEAD`
- `=======`
- `>>>>>>> branch-name`

Keep the correct final code (sometimes a merge of both sides).

## 5) Mark files as resolved and finish

```bash
git add <file1> <file2> ...
git commit -m "Resolve merge conflicts"
```

If you used rebase:

```bash
git rebase --continue
```

## 6) Verify before push

```bash
make conflict-check
make test
```

## 7) Push and update PR

```bash
git push -u origin fix/merge-conflicts
```

Then open/update your PR from this branch.

---

## Quick troubleshooting

- **Still says conflicts after commit?**
  - You likely resolved locally against the wrong base branch. Confirm your PR target and merge/rebase against that exact branch.
- **Accidentally kept markers?**
  - Run `make conflict-check` again and remove all marker blocks.
- **Many files are generated assets** (e.g. minified/css maps):
  - Prefer regenerating from source or accept one side consistently to avoid manual corruption.
