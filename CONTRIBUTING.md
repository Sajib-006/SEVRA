# Contributing

Contributions that make SEVRA easier to reuse across models and workloads are welcome.

1. Fork the repository and create a focused branch.
2. Install the development environment with `pip install -e ".[dev]"`.
3. Add tests for behavior changes.
4. Run `ruff check src tests examples` and `python -m unittest discover -s tests -v`.
5. Open a pull request describing the use case, change, and validation.

Please keep the core package provider-agnostic. Provider-specific clients belong in examples or
optional integrations, and no API keys, raw private prompts, or model checkpoints should be committed.
