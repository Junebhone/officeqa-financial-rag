"""Download the gated OfficeQA data for the configured years.

The Databricks OfficeQA corpus + answer-key CSV are GATED on Hugging Face
(https://huggingface.co/datasets/databricks/officeqa). You must request access
and provide a token via the HF_TOKEN environment variable.

Downloads:
  - officeqa_full.csv                          (answer key)
  - treasury_bulletins_parsed/transformed/treasury_bulletin_YYYY_MM.txt
    for every quarter (03/06/09/12) of each year in config.YEARS that exists.
"""
import os
import sys

import requests

import config

REPO = "databricks/officeqa"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
QUARTERS = ["03", "06", "09", "12"]


def _get(url: str, token: str, dest) -> bool:
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"},
                     allow_redirects=True, timeout=120)  # -L: HF 302s to a CDN
    if r.status_code == 200 and r.content:
        dest.write_bytes(r.content)
        return True
    return False


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        # fall back to a local .hf_token file (gitignored)
        tf = config.ROOT / ".hf_token"
        token = tf.read_text().strip() if tf.exists() else None
    if not token:
        print("ERROR: set HF_TOKEN (request access at "
              "https://huggingface.co/datasets/databricks/officeqa)")
        return 1

    config.CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading answer key...")
    if not _get(f"{BASE}/officeqa_full.csv", token, config.ANSWER_KEY_CSV):
        print("  FAILED — check token / access grant.")
        return 1

    ok = miss = 0
    for year in config.YEARS:
        for q in QUARTERS:
            name = f"treasury_bulletin_{year}_{q}.txt"
            url = f"{BASE}/treasury_bulletins_parsed/transformed/{name}"
            if _get(url, token, config.CORPUS_DIR / name):
                ok += 1
            else:
                miss += 1  # quarter may not exist (e.g. 2025_12)
    print(f"Corpus: {ok} files downloaded, {miss} missing/nonexistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
