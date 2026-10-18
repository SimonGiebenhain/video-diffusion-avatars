"""Add the embedding of a new prompt to weights/text_emb_cache.pt (code: vda/text_cache.py). Needs the text
encoder (python download.py base --with_text_encoder); the released models were trained with one fixed prompt.

    python scripts/add_prompt.py --prompt "<PROMPT>"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vda.text_cache import main  # noqa: E402

if __name__ == "__main__":
    main()
