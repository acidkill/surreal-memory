# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 4.x     | :white_check_mark: |
| 3.x     | :x:                |
| < 3.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in NeuralMemory, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please email: **nhadaututtheky@gmail.com**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

You will receive a response within **48 hours** acknowledging receipt. We aim to release a fix within **7 days** for critical issues.

## Security Measures

NeuralMemory implements several security measures:

- **Parameterized SQL** — All queries use `?` placeholders, never string formatting
- **Path validation** — `Path.resolve()` + `is_relative_to()` on all file access
- **Fernet encryption** — Optional at-rest encryption for sensitive memories
- **Sensitive content detection** — Auto-detect and redact PII, secrets, credentials
- **CORS hardening** — Explicit localhost port list, not wildcard
- **Bind to 127.0.0.1** — Local-only by default, not `0.0.0.0`
- **API key auth** — SHA-256 hashed keys for cloud sync
- **No internal info leaks** — Error messages never expose IDs, paths, or stack traces

## Dependency Updates

We monitor dependencies for known vulnerabilities and update promptly. Security-critical dependency updates are released as patch versions.
