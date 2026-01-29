# Contributing to CORTEX-12

Thank you for your interest in contributing to CORTEX-12! We welcome contributions from the community.

---

## 📋 Table of Contents

* [Code of Conduct](#code-of-conduct)
* [How Can I Contribute?](#how-can-i-contribute)
* [Development Setup](#development-setup)
* [Pull Request Process](#pull-request-process)
* [Coding Standards](#coding-standards)
* [Testing](#testing)
* [Documentation](#documentation)
* [Community](#community)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background or identity.

### Our Standards

**Positive behaviors include**:
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Accepting constructive criticism gracefully
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behaviors include**:
- Harassment, discrimination, or offensive comments
- Trolling, insulting/derogatory comments, or personal attacks
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

### Enforcement

Violations can be reported by opening an issue or contacting the project maintainer. All complaints will be reviewed and investigated promptly and fairly.

---

## How Can I Contribute?

### 🐛 Reporting Bugs

**Before submitting a bug report**:
1. Check the [issue tracker](https://github.com/taylorjohn/cortex-12/issues) for existing reports
2. Verify you're using the latest version
3. Try to reproduce with minimal example

**When submitting**:
- Use a clear, descriptive title
- Provide detailed steps to reproduce
- Include expected vs actual behavior
- Add relevant logs, screenshots, or error messages
- Specify environment (OS, Python version, dependencies)

**Bug Report Template**:
```markdown
**Description**
A clear description of the bug.

**To Reproduce**
1. Go to '...'
2. Run command '....'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., Windows 11]
- Python: [e.g., 3.10.11]
- PyTorch: [e.g., 2.0.1+cpu]
- CORTEX-12 version: [e.g., v12-phase3]

**Additional Context**
Any other relevant information.
```

---

### 💡 Suggesting Enhancements

We welcome feature requests! Consider:

**Areas for Enhancement**:
- 🔬 **New semantic axes** (texture, material, pose, etc.)
- 📊 **Additional benchmarks** (compare to CLIP, JEPA, etc.)
- 🎨 **Visualization tools** (t-SNE, UMAP, activation maps)
- 📚 **Documentation improvements** (tutorials, guides, examples)
- 🧪 **New test cases** (edge cases, validation scripts)
- 🚀 **Performance optimizations** (faster inference, lower memory)

**Enhancement Request Template**:
```markdown
**Feature Description**
Clear description of the proposed feature.

**Motivation**
Why would this be useful? What problem does it solve?

**Proposed Implementation**
(Optional) How might this be implemented?

**Alternatives Considered**
Other solutions you've considered.

**Additional Context**
Any other relevant information.
```

---

### 📝 Improving Documentation

Documentation contributions are highly valuable!

**Documentation needs**:
- Fixing typos or unclear explanations
- Adding code examples
- Creating tutorials
- Translating docs to other languages
- Improving API documentation
- Adding diagrams or visualizations

**How to contribute docs**:
1. Fork the repository
2. Edit files in `docs/` or update docstrings
3. Test that links work and formatting renders correctly
4. Submit pull request with clear description

---

### 🔬 Contributing Research

Have novel ideas for semantic certification or compositional learning?

**Research contributions**:
- Novel certification methods
- New compositional benchmarks
- Theoretical analysis of learned representations
- Applications to new domains (medical imaging, robotics, etc.)
- Comparisons with other interpretable systems

**Process**:
1. Open an issue describing your research idea
2. Get feedback from maintainers and community
3. Implement and document thoroughly
4. Submit pull request with results and analysis

---

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- CPU (no GPU required)

### Installation

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/cortex-12.git
cd cortex-12

# 3. Add upstream remote
git remote add upstream https://github.com/taylorjohn/cortex-12.git

# 4. Create virtual environment
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1
# Linux/Mac
source venv/bin/activate

# 5. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 6. Install development dependencies
pip install -r requirements-dev.txt  # pytest, black, flake8, etc.

# 7. Verify installation
python -m pytest tests/
```

### Development Dependencies

Create `requirements-dev.txt`:
```
pytest>=7.0.0
pytest-cov>=4.0.0
black>=22.0.0
flake8>=5.0.0
mypy>=0.990
pre-commit>=2.20.0
```

---

## Pull Request Process

### 1. Create a Branch

```bash
# Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

**Branch naming**:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `test/` - Test additions/improvements
- `refactor/` - Code refactoring

### 2. Make Your Changes

- Write clean, readable code
- Follow existing code style
- Add tests for new functionality
- Update documentation as needed
- Keep commits focused and atomic

### 3. Test Your Changes

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_certification.py

# Run with coverage
python -m pytest --cov=. tests/

# Run smoke tests
python test_v12_smoke.py

# Run stability tests
python test_v12_compare_stability.py
```

### 4. Commit Your Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add feature: semantic axis discovery"
```

**Commit message guidelines**:
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- First line: brief summary (50 chars or less)
- Blank line, then detailed description if needed
- Reference issues/PRs if applicable (#123)

**Example**:
```
Add automatic semantic axis discovery

Implements unsupervised method to identify which dimensions
encode specific attributes using mutual information.

Closes #123
```

### 5. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 6. Open Pull Request

1. Go to your fork on GitHub
2. Click "Pull Request"
3. Select your branch
4. Fill out PR template (see below)
5. Submit!

**Pull Request Template**:
```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Related Issue
Fixes #(issue number)

## Testing
Describe tests you ran and how to reproduce.

## Checklist
- [ ] My code follows the code style of this project
- [ ] I have updated the documentation accordingly
- [ ] I have added tests to cover my changes
- [ ] All new and existing tests passed
- [ ] My changes generate no new warnings
```

### 7. Code Review

- Maintainers will review your PR
- Address any feedback or requested changes
- Once approved, PR will be merged

---

## Coding Standards

### Python Style Guide

Follow [PEP 8](https://peps.python.org/pep-0008/) with these specifics:

**Code Formatting**:
```bash
# Use Black for formatting
black --line-length 88 .

# Check style with flake8
flake8 --max-line-length 88 --ignore=E203,W503 .
```

**Type Hints**:
```python
def certify_axis(
    model: CortexV12,
    axis_name: str,
    validation_samples: List[Tuple[Image.Image, str]]
) -> Dict[str, Any]:
    """
    Certify a semantic axis using validation data.
    
    Args:
        model: Trained CORTEX-12 model
        axis_name: 'color', 'shape', or 'size'
        validation_samples: List of (image, label) pairs
    
    Returns:
        certificate: Dict with accuracy and centroids
    """
    ...
```

**Docstrings**:
```python
def compute_similarity(
    embedding1: torch.Tensor,
    embedding2: torch.Tensor,
    normalize: bool = True
) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding [D]
        embedding2: Second embedding [D]
        normalize: Whether to L2-normalize before computing similarity
    
    Returns:
        Similarity score in [-1, 1] range
    
    Example:
        >>> emb1 = torch.randn(128)
        >>> emb2 = torch.randn(128)
        >>> sim = compute_similarity(emb1, emb2)
        >>> print(f"Similarity: {sim:.3f}")
    """
    ...
```

**Imports**:
```python
# Standard library
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Third-party
import torch
import numpy as np
from PIL import Image

# Local
from vl_jepa_llm_v12 import CortexV12
from semantic_axes import certify_axis
```

---

## Testing

### Test Structure

```
tests/
├── test_certification.py       # Certification tests
├── test_compositional.py       # Compositional reasoning tests
├── test_zero_shot.py           # Zero-shot generalization tests
├── test_stability.py           # Stability metrics tests
└── fixtures/                   # Test data
    ├── sample_images/
    └── test_certificates/
```

### Writing Tests

```python
import pytest
import torch
from vl_jepa_llm_v12 import CortexV12

@pytest.fixture
def model():
    """Load trained model for testing."""
    return CortexV12.load_checkpoint("brain_vector_v12.pth")

def test_certification_accuracy(model):
    """Test that certification achieves >85% accuracy."""
    from tools.certify_cortex12 import certify_axis
    
    cert = certify_axis(model, "color", validation_samples=100)
    assert cert["accuracy"] > 0.85

def test_zero_shot_generalization(model):
    """Test zero-shot on held-out combinations."""
    from tools.zero_shot_eval import evaluate_zero_shot
    
    accuracy = evaluate_zero_shot(model, test_split=0.9)
    assert accuracy > 0.75
```

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_certification.py

# Specific test function
pytest tests/test_certification.py::test_certification_accuracy

# With coverage
pytest --cov=. --cov-report=html

# Verbose output
pytest -v
```

---

## Documentation

### Documentation Structure

```
docs/
├── README.md                    # Documentation index
├── certification/
│   ├── README.md                # Certification overview
│   ├── methodology.md           # How it works
│   └── examples.md              # Usage examples
├── training/
│   ├── phase1.md                # Phase 1 guide
│   ├── phase2.md                # Phase 2 guide
│   └── phase3.md                # Phase 3 guide
└── use_cases/
    ├── medical_imaging.md       # Medical AI demo
    └── robotics.md              # Robotics applications
```

### Documentation Guidelines

**Markdown formatting**:
- Use clear, descriptive headers
- Include code examples
- Add images/diagrams where helpful
- Link to related documentation
- Keep paragraphs concise

**Code examples**:
- Should be copy-paste runnable
- Include expected output
- Add comments explaining non-obvious parts

**Images**:
- Use descriptive alt text
- Keep file sizes reasonable (<500KB)
- Store in `docs/images/`

---

## Community

### Getting Help

- **GitHub Discussions**: [Ask questions, share ideas](https://github.com/taylorjohn/cortex-12/discussions)
- **GitHub Issues**: [Report bugs, request features](https://github.com/taylorjohn/cortex-12/issues)
- **Documentation**: [Read the docs](https://github.com/taylorjohn/cortex-12/tree/main/docs)

### Staying Updated

- Watch the repository for notifications
- Follow releases for new versions
- Check roadmap for planned features

### Recognition

Contributors will be:
- Listed in AUTHORS.md
- Mentioned in release notes
- Credited in relevant documentation

---

## License

By contributing to CORTEX-12, you agree that your contributions will be licensed under the MIT License.

---

## Questions?

If you have questions about contributing, please:
1. Check existing documentation
2. Search GitHub Issues/Discussions
3. Open a new Discussion if your question hasn't been answered

---

**Thank you for contributing to CORTEX-12!** 🎉
