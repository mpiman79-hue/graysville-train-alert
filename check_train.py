from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from ultralytics import YOLO

# Public Catoosa County page, not the direct Verkada login link
CAMERA_URL = "https://www.catoosacountyga.gov/government/graysville-train-crossing-camera-test"

SCREENSHOT = "camera_screenshot.jpg"
RESULT_FILE = "train_detected.txt"
CONF_FILE = "confidence.txt"


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
            page.goto(CAMERA_URL, wait_until="domcontentloaded", timeout=60000)

            # Give the embedded camera/page time to load
            page.wait_for_timeout(20000)

            # Take screenshot of what GitHub can actually see
            page.screenshot(path=SCREENSHOT, full_page=False)

        except PlaywrightTimeoutError:
            page.screenshot(path=SCREENSHOT, full_page=False)

        finally:
            browser.close()


def detect_train():
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
