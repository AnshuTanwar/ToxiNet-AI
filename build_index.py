"""
ToxiNet AI — Build Similarity Index
Run once before starting the app.

Usage:
    python build_index.py --tox21 tox21.csv
    python build_index.py --tox21 tox21.csv --zinc 250k_rndm_zinc_drugs_clean_3.csv
"""
import argparse, time
from molecular_intelligence import build_similarity_index

parser = argparse.ArgumentParser()
parser.add_argument("--tox21", default="tox21.csv")
parser.add_argument("--zinc",  default=None,
    help="Path to ZINC250k CSV (250k_rndm_zinc_drugs_clean_3.csv)")
args = parser.parse_args()

print("=" * 55)
print("  ToxiNet AI — Similarity Index Builder")
print("=" * 55)
t0 = time.time()
idx = build_similarity_index(args.tox21, args.zinc)
print(f"\nDone in {time.time()-t0:.1f}s  |  {len(idx['entries'])} molecules indexed")
print("\nStart the app:")
print("  streamlit run app.py")
