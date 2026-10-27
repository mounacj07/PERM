import glob
import re

handoffs = glob.glob("HANDOFF_*.md")

for h in handoffs:
    with open(h, "r", encoding="utf-8") as f:
        text = f.read()

    f_uri = "file:" + "///"
    text = re.sub(r"\[([^\]]+)\]\(" + f_uri + r"[a-zA-Z]:/[^)]+/projects/perm/([^)]+)\)", r"[\1](\2)", text)
    text = re.sub(f_uri + r"[a-zA-Z]:/[^\s\)\"\'>]+/projects/perm/", "", text)
    text = re.sub(f_uri + r"[^\s\)\"\'>]+", "", text)
    text = re.sub(r"[a-zA-Z]:/[^\s\)\"\'>]+/projects/perm/", "", text)
    text = re.sub(r"[a-zA-Z]:\\[^\s\)\"\'>]+\\projects\\perm\\", "", text)
    text = re.sub(r"[C]:" + r"\\\.\.\.`", "`<root>`", text)
    text = re.sub("mou" + "na", "<user>", text, flags=re.IGNORECASE)

    with open(h, "w", encoding="utf-8") as f:
        f.write(text)

print("Cleaned all handoffs.")
