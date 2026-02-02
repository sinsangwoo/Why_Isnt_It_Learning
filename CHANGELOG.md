# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-02-03

### Added
- **Phase 1: Modern Foundation**
  - Converted TensorFlow codebase to PyTorch
  - Created installable Python package with `pyproject.toml`
  - Added `GradientAnalyzer` core API
  - Implemented automatic gradient pathology detection
  - Added comprehensive test suite (pytest)
  - Setup GitHub Actions CI/CD
  - Created professional README with project narrative
  - Removed hardcoded Korean fonts, improved internationalization

### Changed
- Repository structure: legacy `Gradient.py` preserved, new code in `src/`
- All visualization now uses system fonts (no Malgun Gothic dependency)

### Technical Details
- Python 3.9+ support
- PyTorch 2.0+ as core dependency
- Type hints throughout codebase
- Linting: ruff, black, mypy

---

## [0.0.1] - 2023 (High School Era)

### Initial Release
- TensorFlow-based gradient visualization
- Manual experiments with sigmoid/tanh/relu
- Basic histogram plotting
- Proof-of-concept: gradient vanishing/exploding reproduction
