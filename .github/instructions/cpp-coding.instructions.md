---
description: "Use when writing or modifying generic C++ code in this repository. Keep the style minimal and consistent until more domain-specific rules are added."
---

- Prefer clear, descriptive names. Use `m_` for private members, `get_`/`set_` for accessors, and `is_` for boolean predicates.
- Keep the API surface explicit: declare public behavior in headers and implement details in the corresponding `.cpp` file.
- Favor RAII and value semantics: use `std::unique_ptr`, `std::vector`, `std::string`, and `std::move` instead of manual ownership.
- Use raw pointers only for non-owning references or API boundaries where the lifetime is explicit.
- Validate inputs early and raise informative exceptions; prefer `pagmo_throw(std::invalid_argument, ...)` for bad values and `pagmo_throw(std::runtime_error, ...)` for setup or runtime failures.
- Keep comments focused on intent, invariants, and non-obvious workarounds; prefer straightforward code over explanatory comments.
- Use small helper functions for validation and conversion instead of embedding too much logic in large methods.
- Keep helper code in `detail` or anonymous namespaces, and avoid leaking internal implementation details into the public API.
- Prefer standard library facilities and clear ownership semantics over custom pointer logic or ad hoc abstractions.
- Match the surrounding style: 4-space indentation, straightforward control flow, and simple, readable construction order.
- Prefer `assert` for internal invariants and exceptions for user-visible validation.
- Keep code easy to inspect and serialize: getters/setters and member state should be predictable and consistent.
- When the container or library defines a non-obvious index type, prefer `decltype(...)` for the loop variable so the loop matches the actual type used by the API. This avoids type mismatches when bounds are expressions or library-specific numeric types, such as Eigen indices.
