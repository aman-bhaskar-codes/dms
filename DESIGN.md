# Sentinel design context

A local driver-monitoring instrument panel for a supervised portfolio demonstration. The primary job is to make tracking quality, current signals, and alerts clear.

See docs/design/implementation.md for Figma link, palette, layout, and behavior. Runtime token owner: ui/web/static/style.css :root. Shared control owner: global button/input/select CSS. Shared feedback and navigation owner: app.js. English; dates use the browser's local timezone.

## Signature
A camera instrument frame with a tracking-status rail; measured values appear below it as instrument readings. Restrained blue slate surfaces and teal for actions; amber and red reserved for alerts.

## Typography and layout
Use local system sans-serif; monospace readings. Large title 32px, section title 20px, body 15px, captions 12px. Main grid 2:1 with 24px gaps. Stack below 980px. No remote font or chart CDN.

## States and accessibility
Visible focus, reduced motion, semantic controls, text status labels, null measurements, stale connection state, persistent validation errors, consistent loading/empty/error feedback. Native selectors and app-owned deletion dialog. Do not render assistant HTML.
