🛠️ Git Professional Cheatsheet

A quick reference for modern and essential Git commands.

---

### ⚙️ Configuration
Set up your Git identity. Do this once per machine.

| Action | Command |
| :--- | :--- |
| Set your name | `git config --global user.name "Your Name"` |
| Set your email | `git config --global user.email "you@example.com"` |

---

### 🚀 Getting Started
| Action | Command |
| :--- | :--- |
| Initialize a new repo | `git init` |
| Clone an existing repo | `git clone <repository-url>` |

---

### Daily Workflow
| Action | Command |
| :--- | :--- |
| Check status of changes | `git status` |
| Stage a specific file | `git add <file>` |
| Stage all changes | `git add .` |
| Commit staged changes | `git commit -m "Your descriptive message"` |
| Change the last commit | `git commit --amend` |

---

### 🌿 Branching (with `git switch`)
Modern commands for navigating and managing branches (Git 2.23+).

| Action | Command |
| :--- | :--- |
| Switch to an existing branch | `git switch <branch-name>` |
| Create and switch to a new branch | `git switch -c <new-branch-name>` |
| Switch back to the previous branch | `git switch -` |
| Merge another branch into current | `git merge <other-branch>` |
| Delete a local branch | `git branch -d <branch-name>` |
| **DANGER:** Delete a local branch (force) | `git branch -D <branch-name>` |

---

### ⏪ Undoing Changes (with `git restore`)
Modern commands for discarding or unstaging work (Git 2.23+).

| Action | Command |
| :--- | :--- |
| Discard unstaged changes in a file | `git restore <file>` |
| Unstage a file (move from staged to unstaged) | `git restore --staged <file>` |
| Restore a file to a specific commit's version | `git restore --source <commit-hash> <file>` |

---

### ⏪ Undoing Commits (with `git reset`)
`git reset` is a powerful command for undoing commits by moving the `HEAD` pointer. It can also modify the staging area and working directory. Use with caution.

**Reset Modes Explained:**

| Mode | Moves `HEAD`? | Updates Staging Area? | Updates Working Directory? | Effect on Changes |
| :--- | :--- | :--- | :--- | :--- |
| `--soft` | Yes | No | No | Undone commits become staged changes. |
| `--mixed` (Default) | Yes | Yes | No | Undone commits become unstaged changes in the working directory. |
| `--hard` | Yes | Yes | Yes | **Destructive.** Undone commits and all associated changes are permanently deleted. |

**Common Commands (to undo the last commit):**

| Action | Command |
| :--- | :--- |
| Un-commit, but keep changes staged | `git reset --soft HEAD~1` |
| Un-commit and un-stage changes | `git reset HEAD~1` |
| **DANGER:** Un-commit and discard all changes | `git reset --hard HEAD~1` |

---

### 🔍 Inspecting History
| Action | Command |
| :--- | :--- |
| View commit history (full) | `git log` |
| View history as a compact graph | `git log --oneline --graph --all` |
| Get short hash of the current commit | `git rev-parse --short HEAD` |

---

### 🌐 Working with Remotes
| Action | Command |
| :--- | :--- |
| Fetch changes and merge them | `git pull origin <branch-name>` |
| Push committed changes to remote | `git push origin <branch-name>` |
| Delete a remote branch | `git push origin --delete <branch-name>` |
| Fetch changes, but don't merge | `git fetch` |

---

### ✨ Special Characters
| Character | Name | Purpose |
| :--- | :--- | :--- |
| `-` | Dash / Tack | Shorthand for "the previous branch" (e.g., `git switch -`). |
| `--` | Double Dash | Separator: tells Git to stop parsing options and treat what follows as file paths. |
| `~` | Tilde | Refers to a parent commit (e.g., `HEAD~1` is one commit before `HEAD`). |
| `^` | Caret | Used for merge commits to specify a parent (e.g., `HEAD^1` is the first parent). |
