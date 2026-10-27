import glob
import re

handoffs = glob.glob("HANDOFF_*.md")
print("Found handoffs:", handoffs)

patterns = [
    r"file:" + r"//",
    r"[a-zA-Z]:\\[a-zA-Z0-9_\\]+",
    r"(?<!https:)(?<!http:)\b[a-zA-Z]:/[a-zA-Z0-9_/]+",
    r"\bC:[\\/]",
    r"\b" + "mou" + "na" + r"\b",
    r"\\Users\\",
    r"/Users/",
]

all_clean = True
for h in sorted(handoffs):
    with open(h, "r", encoding="utf-8") as f:
        content = f.read()
    print(f"\n=== {h} ===")
    found_any = False
    for pat in patterns:
        matches = re.findall(pat, content, re.IGNORECASE)
        if matches:
            found_any = True
            all_clean = False
            print(f"  Pattern '{pat}': {len(matches)} matches (sample: {matches[:3]})")
    if not found_any:
        print("  CLEAN (0 matches)")

print("\nOVERALL CLEAN:", all_clean)
