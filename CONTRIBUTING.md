# Contributing to Watermark Remover Suite

Thank you for your interest in contributing to the Watermark Remover Suite! This document provides guidelines and instructions for contributing to the project.

## Getting Started

### Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/Watermark-Remover-Suite.git
cd Watermark-Remover-Suite
```

2. Set up your development environment:
```bash
make setup
```

This will:
- Create a virtual environment
- Install the package in editable mode
- Install all development dependencies

### Development Workflow

1. Create a new branch for your feature or bugfix:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and ensure code quality:
```bash
# Format your code
make format

# Run linting
make lint

# Run tests
make test
```

3. Commit your changes:
```bash
git add .
git commit -m "Description of your changes"
```

4. Push to your fork and create a Pull Request

## Code Quality Standards

### Testing

- Write tests for new features and bug fixes
- Ensure all tests pass before submitting a PR
- Aim for good test coverage (check with `make test-cov-html`)

```bash
# Run all tests
make test

# Run with coverage report
make test-cov

# Generate HTML coverage report
make test-cov-html
```

### Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting (line length: 100)
- **Ruff**: Fast Python linter
- **mypy**: Static type checking

```bash
# Auto-format code
make format

# Check linting
make lint
```

### Testing Guidelines

- Place tests in the `tests/` directory
- Name test files with the `test_*.py` pattern
- Use descriptive test names: `test_feature_does_something()`
- Use unittest or pytest style tests
- Mock external dependencies when appropriate

## Pull Request Process

1. **Update Documentation**: If you add features, update the README.md and relevant docs
2. **Add Tests**: Ensure your changes are covered by tests
3. **Pass CI Checks**: All GitHub Actions checks must pass
4. **Code Review**: Address any feedback from reviewers
5. **Squash Commits**: Keep your PR history clean

### PR Description Template

```markdown
## Description
Brief description of what this PR does

## Changes
- List of changes made
- Another change

## Testing
- How to test these changes
- What tests were added/modified

## Related Issues
Fixes #123
```

## Continuous Integration

Our CI pipeline runs on every PR and includes:

- **Tests**: Run on Python 3.11 and 3.12
- **Linting**: Black formatting check, Ruff linting
- **Type Checking**: mypy static analysis
- **Coverage**: Code coverage reporting via Codecov
- **Build**: Package distribution verification

All checks must pass before a PR can be merged.

## Reporting Issues

When reporting issues, please include:

- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages or logs
- Screenshots (if applicable)

## Feature Requests

We welcome feature requests! Please:

1. Check if the feature already exists or is planned
2. Open an issue with the "enhancement" label
3. Describe the feature and its use case
4. Discuss implementation approach if relevant

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow the project's coding standards

## Questions?

If you have questions about contributing:

- Open a GitHub Discussion
- Check existing issues and PRs
- Review the documentation in `docs/`

## Development Commands Reference

| Command | Description |
|---------|-------------|
| `make setup` | Set up development environment |
| `make test` | Run all tests |
| `make test-cov` | Run tests with coverage |
| `make test-cov-html` | Generate HTML coverage report |
| `make format` | Auto-format code |
| `make lint` | Run linting checks |
| `make clean` | Clean build artifacts |

Thank you for contributing to make Watermark Remover Suite better!
