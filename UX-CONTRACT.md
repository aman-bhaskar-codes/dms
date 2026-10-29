# Canonical UI behavior

Shared owner: ui/web/static/app.js and style.css.
Navigation: hash routes, semantic buttons, current-page marker, title focus on route change.
Forms: novalidate, visible labels, inline errors, preserved values, busy controls disabled.
Select/Listbox: native select; operating-system popup accepted.
Feedback: one persistent role=status notice; alert severity uses role=alert.
Deletion: dialog with cancel default, keyboard focus trap and restoration.
Offline: connection banner and stale metrics; automatic bounded reconnect.
CRUD: sessions start/stop through one API helper; history reloads after deletion.
Scrollbar: browser-native; stable gutter.

| Capability | Canonical owner | Source of truth | Allowed variants | Verification |
| --- | --- | --- | --- | --- |
| Select/Listbox | Native select | DESIGN.md | OS popup | scripts/browser_check.py |
| Form | app.js API helper | UX-CONTRACT.md | Local / cloud / settings | scripts/browser_check.py |
| Scrollbar | style.css | DESIGN.md | Browser-native | scripts/browser_check.py |
| CRUD | app.js shared actions | API v1 | Session / report | tests/test_runtime.py |

All static buttons start disabled until their external JavaScript handlers are bound. This prevents nonfunctional controls while the script loads and preserves the script-src self policy. Settings inputs remain disabled while saved values load, preventing a late load from overwriting user input.
