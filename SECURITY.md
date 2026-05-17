# Security Policy

## Supported Versions

NIRSPY is pre-alpha. Only the latest commit on `main` receives security fixes.
Once v0.1.0 is released, the latest minor will be supported.

| Version    | Supported |
| ---------- | --------- |
| pre-0.1    | ✅ (latest main only) |

## Reporting a Vulnerability

**Do not open a public issue for security vulnerabilities.**

Use GitHub's private security advisory:
<https://github.com/BrunoFurlanetto/nirspy/security/advisories/new>

Or email the maintainer directly via the address listed on the GitHub profile.

Include:

- Affected version / commit
- Reproduction steps or proof of concept
- Impact assessment (confidentiality / integrity / availability)
- Suggested fix (optional)

You will receive an acknowledgement within 7 days. A patch timeline will be
communicated after triage (typically within 30 days for high-severity issues).

## Threat Model

NIRSPY runs **locally** by default (`127.0.0.1:8050`). It is not designed to be
exposed to the public internet without a reverse proxy and explicit hardening.
Findings tracked in `docs/security/` cover the local-execution threat model
only.
