🐍 A Practical Guide to Python Imports & Modules

This guide explains how Python imports work, focusing on best practices for a project structure like ours. Understanding these concepts is the key to avoiding common `ImportError` issues.

---

## TL;DR: The Two Golden Rules

1.  **Inside the `mlops` package, always use relative imports.**
    - To import from a file in the same directory: `from .module import MyClass`
    - To import from a file in a subdirectory: `from .subpackage.module import MyClass`

2.  **Always run your Python scripts as modules.**
    - Use the `invoke` tasks (e.g., `invoke train`).
    - Or run manually with `uv run python -m mlops.train`.

---

## Understanding the "Why"

To understand the rules, let's look at our project's structure.

```
REPO/
├── src/             <-- This is a "root" on Python's search path.
│   └── mlops/       <-- This is a PACKAGE.
│       ├── __init__.py
│       ├── data/      <-- This is a SUB-PACKAGE.
│       │   ├── __init__.py
│       │   └── dataset.py
│       ├── model.py   <-- This is a MODULE.
│       └── train.py
└── tasks.py
```

- **Module:** A single `.py` file.
- **Package:** A directory with an `__init__.py` file. It groups related modules.
- **`sys.path`:** A list of directories where Python looks for packages to import. Our tooling (`uv run`) automatically adds `src/` to this path.

> **💡 Pro-Tip: When in doubt, check `sys.path`**
> If you get an `ImportError`, you can see exactly where Python is looking. Just add this to your script:
> ```python
> import sys
> from pprint import pprint
> pprint(sys.path)
> ```

---

## The Core Choice: Absolute vs. Relative Imports

| Type | Example (in `train.py`) | When to Use | Why |
| :--- | :--- | :--- | :--- |
| **Relative** | `from .data.dataset import GTSRB` | **Always, for imports *inside* the `mlops` package.** | Makes the package self-contained and easy to move. It clearly shows an internal dependency. |
| **Absolute** | `from mlops.data.dataset import GTSRB` | For imports *outside* the `mlops` package (e.g., in `tests/`). | Explicit and works from anywhere, as long as `src` is on `sys.path`. |

**Our Project's Rule:** We prefer **relative imports** within the `mlops` package to keep it modular.

---

## The #1 Pitfall: How You Run Your Code

The way you execute a script is the most common reason for `ImportError`.

### ❌ The Wrong Way: Running as a simple script

```bash
# From the REPO/ directory...
python src/mlops/train.py
```

- **What happens?** Python doesn't know `train.py` is part of the `mlops` package.
- **The Error:** `ImportError: attempted relative import with no known parent package`. Python sees `from .data...` and has no idea what `.` means.

### ✅ The Right Way: Running as a module

```bash
# From the REPO/ directory...
uv run python -m mlops.train
```

- **What happens?** The `-m` flag tells Python: "Look for a package named `mlops` and run its `train` module." Python understands the full package structure.
- **The Result:** Success! The `.` in `from .data...` correctly resolves to the `mlops` package.

---

## Final Checklist

- [ ] Is my code inside the `src/mlops` package?
  - **Yes:** Use relative imports (`from .something...`).
- [ ] Am I running a script from the `mlops` package?
  - **Yes:** Use `invoke` or `python -m mlops.myscript`.
- [ ] Am I getting an `ImportError`?
  - **Action:** Double-check the two points above. Then, print `sys.path` to see where Python is looking.
