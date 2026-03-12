<!--
SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>

SPDX-License-Identifier: MPL-2.0
-->
# Lessons Learned

- Always run tests and predefined pre-commit hooks before committing.
- Ensure line endings remain as `\n` (LF) instead of `\r\n` (CRLF) for consistency in shared repositories.
- Keep `.gitignore` clean from accidentally committed local files (like `dco_check.txt`).
- Use `git commit -s` consistently for DCO compliance.
