import requests
import re
from bs4 import BeautifulSoup
from rich.console import Console
from rich.prompt import Prompt
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import face_recognition
import pickle
from selenium import webdriver

from selenium.webdriver.common.by import By

from selenium.common.exceptions import WebDriverException, TimeoutException
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary

from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as FirefoxService
import time

class FacebookIdChecker:
    def __init__(self, max_threads=10):
        self.base_url = "https://www.facebook.com/profile.php?id={}"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        self.console = Console()
        self.max_threads = max_threads
        self.output_file_path = "valid_profiles.txt"
        self.image_save_folder = "profile_faces"
        os.makedirs(self.image_save_folder, exist_ok=True)
        os.makedirs("database", exist_ok=True)
        # self.start_id = 1150865029
        # self.end_id = 1350865029
        self.start_id = 0
        self.end_id = 0
        
    def _fetch_soup_with_selenium(self, url):
     options = FirefoxOptions()
     options.add_argument("--headless")
     options.set_preference("intl.accept_languages", "en-US, en")
     options.set_preference("general.useragent.override", self.headers["User-Agent"])

    # Set the path to the Firefox binary explicitly
     options.binary_location = "/snap/firefox/current/usr/lib/firefox/firefox" 

     try:
        driver = webdriver.Firefox(service=FirefoxService(), options=options)
        driver.set_page_load_timeout(15)
        driver.get(url)
        time.sleep(3)
        html = driver.page_source
        driver.quit()
        return BeautifulSoup(html, "html.parser")
     except (WebDriverException, TimeoutException) as e:
        self.console.log(f"[bright_red]Selenium failed for {url}: {e}[/bright_red]")
        return None




    def _get_processed_ids(self):
        processed_ids = set()
        if not os.path.exists(self.output_file_path):
            return processed_ids

        try:
            with open(self.output_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    match = re.search(r"ID: (\d+)", line)
                    if match:
                        processed_ids.add(int(match.group(1)))
        except (IOError, ValueError) as e:
            self.console.print(f"[bold red]Error reading processed IDs: {e}[/bold red]")

        return processed_ids

    def _extract_profile_image_url(self, soup):
        og_image = soup.find("meta", property="og:image")
        if og_image and og_image.get("content"):
            return og_image["content"]
        return None

    def _save_image(self, url, user_id):
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                path = os.path.join(self.image_save_folder, f"{user_id}.jpg")
                with open(path, "wb") as f:
                    f.write(response.content)
                return path
        except Exception as e:
            self.console.log(f"[red]Failed to download image for ID {user_id}: {e}[/red]")
        return None

    def _detect_and_store_face(self, img_path, user_id, name):
        try:
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                encoding = encodings[0]
                safe_name = name.replace(" ", "_")
                with open(os.path.join("database", f"{safe_name}_{user_id}.pkl"), "wb") as f:
                    pickle.dump(encoding, f)
                return True
        except Exception as e:
            self.console.log(f"[red]Face encoding failed for ID {user_id}: {e}[/red]")
        return False

    def _process_id(self, user_id, output_file):
     url = self.base_url.format(user_id)
     soup = self._fetch_soup_with_selenium(url)
 
     if soup:
        try:
            title_tag = soup.find("title")
            if title_tag and title_tag.text:
                title_text = title_tag.text.strip()
                if "facebook" not in title_text.lower() and "log in" not in title_text.lower():
                    full_name = title_text
                    output_line = f"ID: {user_id} | Name: {full_name} | URL: {url}\n"
                    output_file.write(output_line)
                    output_file.flush()

                    image_url = self._extract_profile_image_url(soup)
                    if image_url:
                        img_path = self._save_image(image_url, user_id)
                        if img_path and self._detect_and_store_face(img_path, user_id, full_name):
                            return f"[bold green]\u2713 {full_name} (ID: {user_id}) - Face saved[/bold green]"
                        return f"[yellow]\u2713 {full_name} (ID: {user_id}) - No face[/yellow]"
                    return f"[green]\u2713 {full_name} (ID: {user_id}) - No image[/green]"
        except Exception as e:
            return f"[red]Parsing error for ID {user_id}: {e}[/red]"
     return None


    def get_user_input(self):
     self.console.print("[bold magenta]Welcome to the Improved FB ID Checker with Face Detection![/bold magenta]", justify="center")
     self.console.print("[yellow]Disclaimer: Use responsibly and at your own risk.[/yellow]", justify="center")

     try:
        print("\nSelect search type:")
        print("1: Single ID + range")
        print("2: Full ID Range")
        search_type = input("Your choice [default=2]: ").strip() or "2"

        if search_type == "1":
            single_id = int(input("Enter the starting Facebook profile ID: ").strip())
            num_check = int(input("How many IDs to check from start? ").strip())
            self.start_id = single_id
            self.end_id = single_id + num_check - 1
        else:
            self.start_id = int(input("Enter the start of the ID range: ").strip())
            self.end_id = int(input("Enter the end of the ID range: ").strip())

        if self.start_id >= self.end_id:
            self.console.print("[bold red]Start ID must be less than the end ID.[/bold red]")
            return False

        out_file = input(f"Enter output file path [default={self.output_file_path}]: ").strip()
        if out_file:
            self.output_file_path = out_file

        thread_input = input(f"Enter number of threads [default={self.max_threads}]: ").strip()
        if thread_input:
            self.max_threads = int(thread_input)

     except (ValueError, TypeError):
        self.console.print("[bold red]Invalid input. Please enter valid numbers.[/bold red]")
        return False
     except KeyboardInterrupt:
        self.console.print("\n[bold yellow]Operation cancelled by user.[/bold yellow]")
        return False

     return True


    def run(self):
        if not self.get_user_input():
            return

        processed_ids = self._get_processed_ids()
        if processed_ids:
            self.console.print(f"[bold yellow]Resuming from {len(processed_ids)} previously checked IDs.[/bold yellow]")

        ids_to_check = [i for i in range(self.start_id, self.end_id + 1) if i not in processed_ids]

        if not ids_to_check:
            self.console.print("[bold green]All IDs in the range have already been checked.[/bold green]")
            return

        self.console.print(f"[cyan]Scanning {len(ids_to_check)} IDs using {self.max_threads} threads...[/cyan]")

        progress_columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TextColumn("\u2022"),
            TimeRemainingColumn(),
            TextColumn("\u2022"),
            TimeElapsedColumn(),
        ]

        try:
            with Progress(*progress_columns, console=self.console) as progress:
                task = progress.add_task("[yellow]Checking...", total=len(ids_to_check))

                with open(self.output_file_path, "a", encoding="utf-8") as output_file:
                    with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                        futures = {executor.submit(self._process_id, user_id, output_file): user_id for user_id in ids_to_check}
                        for future in futures:
                            result = future.result()
                            if result:
                                self.console.print(result)
                            progress.update(task, advance=1)

        except KeyboardInterrupt:
            self.console.print("\n[bold red]Process interrupted by user.[/bold red]")
        except Exception as e:
            self.console.print(f"\n[bold red]Unexpected error: {e}[/bold red]")
        finally:
            self.console.print(f"\n[bold magenta]Scan complete. Results saved to '{self.output_file_path}'[/bold magenta]")


if __name__ == "__main__":
    checker = FacebookIdChecker()
    checker.run()