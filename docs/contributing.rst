Contributing to vLLM-Watermark
==============================

Thank you for your interest in contributing to vLLM-Watermark! This document provides guidelines for contributing to the project.

Getting Started
---------------

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a virtual environment** and install dependencies
4. **Create a feature branch** for your changes

Development Setup
-----------------

.. code-block:: bash

   # Clone your fork
   git clone https://github.com/yourusername/vllm-watermark.git
   cd vllm-watermark

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install in development mode
   pip install -e ".[dev]"

   # Install pre-commit hooks
   pre-commit install

Code Style
----------

We use several tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **mypy** for type checking
- **pytest** for testing

Run these before committing:

.. code-block:: bash

   # Format code
   black vllm_watermark tests examples

   # Sort imports
   isort vllm_watermark tests examples

   # Type checking
   mypy vllm_watermark

   # Run tests
   pytest tests/

Testing
-------

Write tests for new features and ensure all tests pass:

.. code-block:: bash

   # Run all tests
   pytest tests/

   # Run with coverage
   pytest tests/ --cov=vllm_watermark

   # Run specific test file
   pytest tests/test_core.py

Documentation
-------------

Update documentation for new features:

1. **Docstrings**: Add comprehensive docstrings to new functions/classes
2. **Examples**: Add examples to the appropriate documentation files
3. **API Reference**: Ensure new APIs are properly documented

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd docs
   make html
   # Open _build/html/index.html in your browser

Pull Request Process
--------------------

1. **Create a feature branch** from `main`
2. **Make your changes** following the coding standards
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Run all checks** locally
6. **Submit a pull request** with a clear description

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

- Use clear, descriptive commit messages
- Include tests for new functionality
- Update documentation for API changes
- Ensure all CI checks pass
- Add a description of your changes

Issue Reporting
---------------

When reporting issues, please include:

- **Environment details** (OS, Python version, dependencies)
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Error messages** and stack traces
- **Minimal example** code if applicable

Feature Requests
----------------

For feature requests:

- **Describe the feature** clearly
- **Explain the use case** and benefits
- **Consider implementation** complexity
- **Check if similar features** already exist

Code of Conduct
---------------

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.

Contact
-------

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and general discussion
- **Email**: For sensitive matters, contact the maintainers directly

Thank you for contributing to vLLM-Watermark!