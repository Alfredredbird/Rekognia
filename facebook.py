import os
import json
import time
import random
import requests
import face_recognition
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import tempfile
import shutil
from colorama import *
from concurrent.futures import ThreadPoolExecutor, as_completed

# === CONFIG ===
START_ID = 100012345678900
NUM_IDS = 500
OUTPUT_JSON = "valid_profiles.json"
IMAGE_DIR = "downloaded_faces"
MAX_THREADS = 5  # Adjust based on system capabilities

# === Setup Chrome Driver ===
def create_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    temp_profile_dir = tempfile.mkdtemp()
    options.add_argument(f"--user-data-dir={temp_profile_dir}")
    try:
        driver = webdriver.Chrome(options=options)
        return driver, temp_profile_dir
    except Exception as e:
        print("[!] Failed to launch Chrome:", e)
        shutil.rmtree(temp_profile_dir)
        return None, None

# === Face detection ===
def is_face_detected(image_bytes):
    try:
        img = face_recognition.load_image_file(BytesIO(image_bytes))
        faces = face_recognition.face_locations(img)
        return len(faces) > 0
    except Exception as e:
        print(Fore.RED + f"[!] Face detection error: {e}" + Fore.RESET)
        return False

def scrape_profile(fb_id):
    driver, temp_profile_dir = create_driver()
    if not driver:
        return None

    profile_url = f"https://www.facebook.com/profile.php?id={fb_id}"
    try:
        driver.get(profile_url)
        time.sleep(random.uniform(2, 4))

        images = driver.find_elements(By.TAG_NAME, "img")
        candidate_url = None
        max_size = 0

        for img in images:
            try:
                src = img.get_attribute("src")
                if not src or "scontent" not in src:
                    continue
                width = int(img.get_attribute("width") or 0)
                height = int(img.get_attribute("height") or 0)
                area = width * height
                if area > max_size:
                    candidate_url = src
                    max_size = area
            except:
                continue

        if not candidate_url:
            print(Fore.RED + f"[-] No suitable image found for {fb_id}" + Fore.RESET)
            return None

        response = requests.get(candidate_url, timeout=10)
        if response.status_code != 200:
            print(f"[!] Failed to fetch image for {fb_id}")
            return None

        if not is_face_detected(response.content):
            print(f"[-] No face found for {fb_id}")
            return None

        os.makedirs(IMAGE_DIR, exist_ok=True)
        img_path = os.path.join(IMAGE_DIR, f"{fb_id}.jpg")
        with open(img_path, "wb") as f:
            f.write(response.content)

        print(Fore.GREEN + f"[+] Face found! Saved {img_path}" + Fore.RESET)
        return {
            "id": str(fb_id),
            "profile_url": profile_url,
            "image_url": candidate_url
        }

    except Exception as e:
        print(f"[!] Error for ID {fb_id}: {e}")
        return None
    finally:
        driver.quit()
        shutil.rmtree(temp_profile_dir, ignore_errors=True)

# === Main function with threading ===
def main():
    results = []

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {executor.submit(scrape_profile, START_ID + i): START_ID + i for i in range(NUM_IDS)}

        for future in as_completed(futures):
            fb_id = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"[!] Thread error for ID {fb_id}: {e}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=4)

    print(f"[✓] Done. {len(results)} valid profiles saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
