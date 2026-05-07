from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from ultralytics import YOLO

# Start on the Catoosa County public page
COUNTY_PAGE_URL = "https://www.catoosacountyga.gov/government/graysville-train-crossing-camera-test"

SCREENSHOT = "camera_screenshot.jpg"
RESULT_FILE = "train_detected.txt"
CONF_FILE = "confidence.txt"
STATUS_FILE = "page_status.txt"


def take_screenshot():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])

        page = browser.new_page(
            viewport={"width": 1280, "height": 720},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )

        try:
            # Open the public county page first
            page.goto(COUNTY_PAGE_URL, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(5000)

            # Click the county's "Enlarged View" link, which should open the actual camera
            with page.expect_popup(timeout=15000) as popup_info:
                page.get_by_text("Enlarged View").click()

            camera_page = popup_info.value
            camera_page.wait_for_load_state("domcontentloaded", timeout=60000)

            # Give the camera stream time to load
            camera_page.wait_for_timeout(25000)

            # Save what GitHub actually sees
            camera_page.screenshot(path=SCREENSHOT, full_page=False)

            page_text = camera_page.locator("body").inner_text(timeout=10000).lower()

            if "log in" in page_text or "enter the email address" in page_text:
                Path(STATUS_FILE).write_text("blocked_login_page")
            else:
                Path(STATUS_FILE).write_text("camera_page_loaded")

        except PlaywrightTimeoutError:
            page.screenshot(path=SCREENSHOT, full_page=False)
            Path(STATUS_FILE).write_text("timeout")

        except Exception as e:
            page.screenshot(path=SCREENSHOT, full_page=False)
            Path(STATUS_FILE).write_text(f"error: {e}")

        finally:
            browser.close()


def detect_train():
    status = Path(STATUS_FILE).read_text() if Path(STATUS_FILE).exists() else "unknown"

    # Do not analyze a login page
    if status in ["blocked_login_page", "timeout"] or status.startswith("error"):
        Path(RESULT_FILE).write_text("false")
        Path(CONF_FILE).write_text("0.0")
        return

    model = YOLO("yolov8n.pt")
    results = model.predict(SCREENSHOT, conf=0.35, verbose=False)

    best_conf = 0.0
    train_found = False

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id].lower()
            confidence = float(box.conf[0])

            if label == "train":
                train_found = True
                best_conf = max(best_conf, confidence)

    Path(RESULT_FILE).write_text("true" if train_found else "false")
    Path(CONF_FILE).write_text(str(round(best_conf, 3)))


if __name__ == "__main__":
    take_screenshot()
    detect_train()
