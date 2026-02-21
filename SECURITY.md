# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 0.7.x   | ✅         |
| < 0.7   | ❌         |

## Reporting a Vulnerability

Please report security vulnerabilities through
[GitHub's private vulnerability reporting](https://github.com/Slush97/scry/security/advisories/new).

Do **not** open a public issue for security-sensitive bugs.

## Scope

The following areas are in scope for security reports:

- **Terminal escape sequence parsing** (`transport/probe.rs`, protocol backends)
- **GPU shader input validation** (`rasterize/shaders/`, `sdf/shaders/`)
- **POSIX shared memory handling** (`transport/kitty_shm.rs`, feature `shm`)
- **Image data processing** (PNG encoding, RGBA buffer handling)

## Response

We aim to acknowledge reports within 48 hours and provide a fix or
mitigation within 7 days for confirmed vulnerabilities.
