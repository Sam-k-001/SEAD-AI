# fix_clean.py — fixes the enron loader in clean.py

with open('preprocessing/clean.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and show current enron function
start = content.find('def load_enron')
end = content.find('\ndef ', start + 1)
print("CURRENT ENRON FUNCTION:")
print(content[start:end])
print("\n" + "="*50)

# Fix the enron loader
old_line = "    df = pd.read_csv(path, usecols=[\"message\"], nrows=15000, encoding=\"latin-1\")\n    df.columns = [\"text\"]"

new_lines = """    df = pd.read_csv(path, nrows=15000, encoding="latin-1")
    if "body" in df.columns:
        df = df[["body"]].copy()
    elif "message" in df.columns:
        df = df[["message"]].copy()
    else:
        df = df[[df.columns[0]]].copy()
    df.columns = ["text"]"""

if old_line in content:
    content = content.replace(old_line, new_lines)
    with open('preprocessing/clean.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("SUCCESS! clean.py fixed!")
else:
    print("Pattern not found!")
    print("Looking for usecols line...")
    idx = content.find("usecols")
    print(content[idx-50:idx+100])