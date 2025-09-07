# Useful Commands

## Create Archive

Create a tar.gz archive of the repository with dataset included:

```bash
tar --exclude='*.pyc' --exclude='__pycache__' --exclude='.git' --exclude='env' --exclude='venv' --exclude='*.tar.gz' --exclude='results' --exclude='*.pkl' --exclude='*.pth' --exclude='*.h5' -czf cluster-shuffling-fl.tar.gz .
```

This command:
- Includes the dataset directory for a self-contained archive
- Excludes Python cache files (*.pyc, __pycache__)
- Excludes virtual environments (env, venv)
- Excludes Git repository (.git)
- Excludes previous archives (*.tar.gz)
- Excludes results directory
- Excludes model files (*.pkl, *.pth, *.h5)