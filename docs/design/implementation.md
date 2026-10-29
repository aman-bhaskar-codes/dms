# Sentinel V6 design specification

Figma: https://www.figma.com/design/xSyBgQHfVPA0M9Fm9A9Wqu

Six editable screens: Live, Setup, Assistant, History, Report, Settings. Figma contains synthetic examples only.
Implementation uses a local system sans-serif stack (SF Pro on macOS), with monospace instrument values. Figma Inter is a portable specification substitute, not a bundled runtime font.

## Tokens
Background #0B141C; panel #15232F; text #DDE8F0; muted #8DABBD; accent #71D8CF; warning #FFD083; danger #FF9292. Grid: 8px; panel radius: 12px; content max-width: 1480px.

## Behavior
Camera and alert are primary. Setup errors provide a next action. No invented heartbeat, forecast or confidence. Unavailable readings show an em dash and reason. All replay views explicitly say synthetic. Six views share navigation, controls, tokens and error handling. Inline status uses role=status; critical messages use role=alert. Settings apply next session. Native selects deliberately retain OS-owned popup geometry. Session deletion uses an app dialog. A print stylesheet hides navigation and controls.
