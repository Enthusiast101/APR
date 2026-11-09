"""
Resumable CIFAR-10 downloader for Windows/Python.

Downloads the official CIFAR-10 Python archive using HTTP Range requests,
so an interrupted download can resume from the existing partial file.

Usage:
    python download_cifar10_resume.py

After download, the script verifies the expected archive size and extracts it.
Then load it with:
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

Note: Resume works only if the server/proxy supports HTTP Range requests.
"""

from pathlib import Path
import os
import sys
import time
import urllib.error
import urllib.request
import tarfile

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
EXPECTED_SIZE = 170_498_071
CHUNK_SIZE = 1024 * 1024  # 1 MiB
RETRIES = 10
RETRY_DELAY_SECONDS = 3

# Use Keras's usual Windows cache location.
CACHE_DIR = Path.home() / ".keras" / "datasets"
PART_PATH = CACHE_DIR / "cifar-10-python.tar.gz.part"
ARCHIVE_PATH = CACHE_DIR / "cifar-10-python.tar.gz"


def download_resumable():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # If a complete archive already exists, don't download it again.
    if ARCHIVE_PATH.exists() and ARCHIVE_PATH.stat().st_size == EXPECTED_SIZE:
        print(f"Complete archive already exists: {ARCHIVE_PATH}")
        return ARCHIVE_PATH

    # Continue from bytes already saved in .part.
    offset = PART_PATH.stat().st_size if PART_PATH.exists() else 0
    if offset > EXPECTED_SIZE:
        print("Partial file is larger than expected; restarting partial download.")
        PART_PATH.unlink()
        offset = 0

    for attempt in range(1, RETRIES + 1):
        try:
            headers = {"User-Agent": "Python resumable CIFAR-10 downloader"}
            if offset > 0:
                headers["Range"] = f"bytes={offset}-"

            request = urllib.request.Request(URL, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                status = getattr(response, "status", response.getcode())

                # If server ignores Range, avoid appending the full file to partial data.
                if offset > 0 and status != 206:
                    print("Server did not honor Range request; restarting from byte 0.")
                    response.close()
                    PART_PATH.unlink(missing_ok=True)
                    offset = 0
                    request = urllib.request.Request(
                        URL, headers={"User-Agent": "Python resumable CIFAR-10 downloader"}
                    )
                    with urllib.request.urlopen(request, timeout=60) as response2:
                        write_response(response2, "wb", 0)
                else:
                    mode = "ab" if offset > 0 else "wb"
                    write_response(response, mode, offset)

            # If file still isn't complete, retry from the current byte count.
            if PART_PATH.exists():
                offset = PART_PATH.stat().st_size
            if offset == EXPECTED_SIZE:
                PART_PATH.replace(ARCHIVE_PATH)
                print(f"\nDownload complete: {ARCHIVE_PATH}")
                return ARCHIVE_PATH
            if offset > EXPECTED_SIZE:
                raise RuntimeError(
                    f"Downloaded {offset:,} bytes, more than expected {EXPECTED_SIZE:,}."
                )
            print(
                f"\nConnection ended early; saved {offset:,}/{EXPECTED_SIZE:,} bytes. "
                f"Retrying in {RETRY_DELAY_SECONDS}s (attempt {attempt}/{RETRIES})..."
            )

        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
            offset = PART_PATH.stat().st_size if PART_PATH.exists() else 0
            print(
                f"\nAttempt {attempt}/{RETRIES} failed: {exc}\n"
                f"Saved progress: {offset:,}/{EXPECTED_SIZE:,} bytes."
            )

        if attempt < RETRIES:
            time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError(
        f"Could not finish after {RETRIES} attempts. Partial data remains at:\n{PART_PATH}\n"
        "Run this script again later; it will resume from the saved byte offset."
    )


def write_response(response, mode, starting_offset):
    total = response.headers.get("Content-Length")
    total = int(total) if total and total.isdigit() else None
    downloaded = starting_offset
    last_print = time.time()

    with open(PART_PATH, mode) as output:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)

            now = time.time()
            if now - last_print >= 0.5:
                pct = min(downloaded / EXPECTED_SIZE * 100, 100)
                print(
                    f"\rDownloaded {downloaded:,}/{EXPECTED_SIZE:,} bytes ({pct:.1f}%)",
                    end="",
                    flush=True,
                )
                last_print = now

        output.flush()
        os.fsync(output.fileno())


def main():
    archive = download_resumable()

    # Verify the archive can be opened before extraction.
    print("Verifying archive...")
    try:
        with tarfile.open(archive, "r:gz") as tar:
            tar.getmembers()
    except (tarfile.TarError, OSError) as exc:
        raise RuntimeError(
            f"Archive verification failed: {exc}. Delete {archive} and rerun."
        ) from exc

    print("Extracting CIFAR-10 archive into the Keras dataset cache...")
    with tarfile.open(archive, "r:gz") as tar:
        # The official archive contains the top-level directory cifar-10-batches-py.
        tar.extractall(path=CACHE_DIR, filter="data")
    print("Extraction complete.")
    print(f"Cache directory: {CACHE_DIR}")
    print("Now try: from keras.datasets import cifar10; cifar10.load_data()")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
