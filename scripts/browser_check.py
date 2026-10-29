"""Run against a local server: python scripts/browser_check.py [base-url]."""

import json
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

base = (
    [arg for arg in sys.argv[1:] if not arg.startswith("--")][0]
    if any(not arg.startswith("--") for arg in sys.argv[1:])
    else "http://127.0.0.1:8000"
)
save_screenshot = "--save-screenshot" in sys.argv
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(base)
    page.get_by_role("heading", name="Your drive, in view.").wait_for()
    page.get_by_role("button", name="Start replay", exact=True).click()
    page.get_by_text("Synthetic replay", exact=True).wait_for()
    page.wait_for_timeout(1400)
    assert page.locator("#state").inner_text() == "Monitoring"
    if save_screenshot:
        page.screenshot(
            path=str(Path(__file__).resolve().parents[1] / "docs/design/dashboard.png"),
            full_page=True,
        )
    page.get_by_role("button", name="Stop session", exact=True).click()
    page.get_by_role("button", name="History", exact=True).click()
    page.get_by_role("button", name="Open report").first.click()
    page.get_by_role("heading", name="Session report").wait_for()
    assert "Tracking coverage" in page.locator("#report-content").inner_text()
    page.get_by_role("img", name="Recorded score trend").wait_for()
    page.get_by_role("button", name="Assistant", exact=True).click()
    page.get_by_role("button", name="Get local status", exact=True).click()
    page.locator("#assistant-answer").get_by_text(
        "Monitoring has no valid current measurement.", exact=False
    ).wait_for()
    page.get_by_label("Local question", exact=True).fill("How should calibration use lighting?")
    page.get_by_role("button", name="Ask locally", exact=True).click()
    page.locator("#citations details").first.wait_for()
    page.locator("#citations summary").first.click()
    page.locator("#citations details[open] p").wait_for()
    page.get_by_role("button", name="Reload assistant settings", exact=True).click()
    page.get_by_text(
        "Assistant settings reloaded. Your monitoring session is unchanged.", exact=True
    ).wait_for()
    page.get_by_role("button", name="Settings", exact=True).click()
    page.get_by_role("heading", name="Make DMS yours.").wait_for()
    original_settings = page.request.get(base + "/api/v1/settings").json()
    page.get_by_label("Warning threshold").fill("90")
    page.get_by_label("Critical threshold").fill("70")
    page.get_by_role("button", name="Save settings", exact=True).click()
    page.locator("#settings-message").get_by_text(
        "Invalid settings or threshold order", exact=True
    ).wait_for()
    for key, value in original_settings.items():
        if key != "audio_enabled":
            page.locator("#" + key).fill(str(value))
        else:
            page.locator("#" + key).set_checked(value)
    page.get_by_role("button", name="Save settings", exact=True).click()
    page.get_by_text("Settings saved. They apply to your next session.", exact=True).wait_for()
    page.get_by_role("button", name="History", exact=True).click()
    before = page.locator(".history-row").count()
    page.locator(".history-row").first.get_by_role("button", name="Delete", exact=False).click()
    page.get_by_role("dialog").wait_for()
    page.get_by_role("button", name="Cancel", exact=True).click()
    assert page.locator(".history-row").count() == before
    page.locator(".history-row").first.get_by_role("button", name="Delete", exact=False).click()
    page.get_by_role("button", name="Delete session", exact=True).click()
    page.get_by_text("Session and its measurements deleted.", exact=True).wait_for()
    assert page.locator(".history-row").count() == before - 1
    page.get_by_role("button", name="Live", exact=True).click()

    for width, height in [(1280, 800), (640, 900), (640, 400)]:
        page.set_viewport_size({"width": width, "height": height})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.emulate_media(reduced_motion="reduce")
    page.keyboard.press("Tab")
    assert page.evaluate("document.activeElement.tagName") in ("BUTTON", "A", "SELECT", "INPUT")
    assert not errors, errors
    print(
        json.dumps(
            {
                "browser": "chromium",
                "checks": "session, report, local assistant, invalid settings, responsive, keyboard, reduced motion",
                "console_errors": errors,
            }
        )
    )
    browser.close()
