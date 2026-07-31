import os

# Directories
#
# Two corpora, two indexes. `chroma_db/` is built from the SRD and committed, so
# a clone retrieves out of the box. `chroma_db_full/` is built from your own
# copies of the commercial rulebooks and is gitignored — broader coverage, not
# redistributable.
CHROMA_DB_DIRECTORY = os.environ.get("DND_CHROMA_DIR", "chroma_db")
FULL_CHROMA_DB_DIRECTORY = "chroma_db_full"

# The vendored SRD 5.1 corpus (CC-BY-4.0). Ships with the repository.
SRD_DIRECTORY = "corpus/srd"

DOCUMENT_PATHS = {
    "Dungeon Master's Guide": "Documents/Dungeon_Masters_Guide_5e.pdf",
    "Player's Handbook": "Documents/Players_Handbook_5e.pdf",
    "Monster Manual": "Documents/Monster_Manual_5e.pdf",
}

# Embedding Model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
