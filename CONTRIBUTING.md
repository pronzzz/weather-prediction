## CONTRIBUTING.md

# Contributing to Weather-Prediction

First off, thank you for considering a contribution! 🎉 We welcome all improvements—bug fixes, enhancements, documentation, or tests. Please follow this guide to make the process smooth and efficient.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)  
2. [Getting Started](#getting-started)  
3. [How to Submit Changes](#how-to-submit-changes)  
4. [Style Guides](#style-guides)  
   - [Commit Messages](#commit-messages)  
   - [Code Formatting](#code-formatting)  
5. [Reporting Bugs](#reporting-bugs)  
6. [Suggesting Enhancements](#suggesting-enhancements)  
7. [Writing Tests](#writing-tests)  
8. [Pull Request Process](#pull-request-process)  
9. [Contact](#contact)  

---

## Code of Conduct

This project and everyone participating in it is governed by the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/) (Version 2.1) citeturn0search7. By participating, you are expected to uphold this code.

---

## Getting Started

1. **Fork** the repo and clone your fork:
   ```bash
   git clone https://github.com/your-username/weather-prediction.git
   cd weather-prediction
   ```
2. **Create a new branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run tests** to verify your environment:
   ```bash
   pytest --maxfail=1 --disable-warnings -q
   ```

---

## How to Submit Changes

- **Branch naming**: use `feature/`, `fix/`, or `docs/` prefixes.  
- **Commit often**, with focused changes per commit.  
- **Rebase** on `main` before opening a PR to keep history clean.

---

## Style Guides

### Commit Messages

Follow the [Angular convention](https://www.conventionalcommits.org/en/v1.0.0/) for commit messages citeturn0search1:
```
<type>(<scope>): <short summary>
```
- **type**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, or `chore`  
- **scope**: area of the code (e.g., `train`, `api`, `gradcam`)  
- **short summary**: max 50 chars  

### Code Formatting

- **Python**: follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines citeturn0search0.  
- **Notebooks**: clear outputs, use descriptive titles, and run all cells top-to-bottom before committing.

---

## Reporting Bugs

Please search existing issues first. If none match, open a new issue and include:

1. **Description** of the problem.  
2. **Steps to reproduce**, with code snippets if possible.  
3. **Expected vs. actual behavior**.  
4. **Environment details**: OS, Python version, dependencies.

---

## Suggesting Enhancements

Feature requests are welcome! To propose an enhancement:

1. Open an issue with the “enhancement” label.  
2. Describe the use case and benefit.  
3. Optionally, submit a PR with a prototype.

---

## Writing Tests

- Use `pytest`.  
- Aim for **> 80%** code coverage.  
- Place tests in the `tests/` directory, mirroring the module structure.  
- Name test functions clearly (e.g., `test_train_model_saves_checkpoint()`).

---

## Pull Request Process

1. Fork → branch → commit → push.  
2. Open a PR against `main`.  
3. Ensure all checks pass (lint, tests).  
4. Fill out the PR template, linking related issues.  
5. Address review feedback promptly.

---

## Contact

For questions or additional guidance, reach out to the maintainers:

- **Lead Maintainer**: @pronzzz  
- **Email**: pranavdofficial@gmail.com

Thank you for helping make Weather-Prediction better!
