# C++ Project Prompt Template

## Coding Style & Naming Conventions

- **Namespace**: `hellofem::`
- **Limited OOP**: No hierarchical inheritance. Use OOP features like inheritance and polymorphism only for sharing interface. Use DOP for most cases.
- **PIMPL pattern**: PIMPL is considerable for decoupling and for reducing compilation dependency.
- **No dynamic_cast**: Always use virtual `configure()` method for parameter injection
- **No Backward Compatibility**: Strictly forbid anything remained for backward-compatibility.
- **No Cross Dependency**: Any two files or modules/libs shall not rely on each other. Dependencies shall only happen on single direction.
- **Do not overuse comments**: Code should be as self-explanatory as possible through naming, interface design... Always add comments **only when and where necessary**, mainly in the following cases:
  - There are implicit data contracts, preconditions, side effects, etc.
  - The function name and parameter names are insufficient to clearly convey the meaning (for example, what unit a piece of data is measured in).
  - The data layout follows a specific contract (for example, the meaning of each dimension in a multi-dimensional array, or the meaning of each offset in a flattened one-dimensional array).
  - There is logic that is difficult to understand directly from the code itself (for example, the use of uncommon formulas or bit manipulation techniques).
- **Avoid using try-catch and C++ std exceptions outside the io module**: Using try-catch in lower-level parts of the system may hide real errors, introduce performance noise, and make testing more difficult.
- **Always centralize default values or fallback values in one place instead of scattering them throughout the codebase**: Except at the I/O layer, fallback handling should not be performed without a clear reason or business requirement. Invalid data should directly trigger an error path.
- **Logging is mandatory**: Logs should always follow a consistent format and use appropriate levels. The default level should usually be INFO, while DEBUG-level logs should generally only be enabled for debugging. Avoid excessive logging; except for DEBUG logs, other logs should never be emitted inside hot loops.

## Simplify the code

- No checks for nonexistent (or repeatedly checked) edge cases. No forward-compatibility mechanisms for requirements that do not currently exist. No unnecessary parameterization unless required.
- If a function, class, or abstraction is unused or used in only one place, meaning it provides no actual reuse value, it should be eliminated.
- If duplicated boilerplate code appears more than three times, extract it into a common function.
- A function should generally not have more than three parameters. If a function violates this rule, it may need to be curried.
- Do not sacrifice performance, readability, or testability merely to reduce code length.
- Backward compatibility may only be maintained for public interface, but never for interior library or test uses.

## Activity Tracking (Required)

- Summaries should include what changed, files touched, and any notable decisions.
- Use the scratchpad tool for follow-ups or TODOs discovered during work.

## Build, Test, and Development Commands

- **Build Config**: C++20, MSVC `/W4 /WX /permissive- /utf-8 /bigobj`, else `-Werror -Wall -Wextra -Wpedantic`.

## Testing Guidelines

- **Test utilities**: `pytest` and `ctest` for testing
- **Enforce TDD for every behavior change**: follow `red -> green -> refactor`.
- **Start with verifiable baseline**: run the relevant existing tests before edits, and record the exact command + outcome in the PR/commit notes.
- **Test updates goes first**: Add or update a failing test first that reproduces the bug or captures the new requirement; implement code only after the test fails for the expected reason.
- **Keep tests green**: Never commit if there are failing tests.
- **Regression test**: Any bug fix must include a regression test that fails before the fix and passes after it.

## Commit & Pull Request Guidelines

- Use Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`) and keep messages imperative.
- PRs: include a short summary, exact test command(s) run, and call out any changes to on-disk memory formats or `qmd` behavior.
