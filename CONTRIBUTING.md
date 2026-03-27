# Contributing to Filo

Thanks for contributing.

## Quick Start

```bash
cmake --preset linux-debug
cmake --build --preset linux-debug
ctest --preset linux-debug --output-on-failure
```

For thread-safety checks:

```bash
cmake --preset linux-tsan
cmake --build --preset linux-tsan
ctest --preset linux-tsan --output-on-failure
```

## Pull Request Guidelines

- Keep PRs focused on one problem.
- Include a short problem statement and your approach.
- Add or update tests for behavior changes.
- Prefer incremental commits over large unreviewable drops.

## Code Style

- Use modern C++ and keep interfaces explicit.
- Favor readable naming over compact cleverness.
- Add comments only when the intent is not obvious from code.
- Avoid unrelated refactors in feature/bug-fix PRs.

## Areas Where Help Is Valuable

- Provider and protocol integrations
- Routing policies and guardrail behavior
- MCP interoperability and transport hardening
- TUI usability and accessibility
- Test coverage for streaming/tool-calling edge cases

## Reporting Bugs

Open an issue with:
- expected behavior
- actual behavior
- reproduction steps
- platform details (OS, compiler, CMake, provider)

## Questions

If a task is large or ambiguous, open an issue first so we can align on direction before implementation.
