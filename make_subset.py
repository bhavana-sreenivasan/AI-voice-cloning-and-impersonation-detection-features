import os
import random
import shutil
audio_dir = r"C:\Users\bhavana s\Documents\Downloads\AI-voice-cloning-and-impersonation-detection-features-main\data\LA\ASVspoof2019_LA_train\flac"
protocol = r"C:\Users\bhavana s\Documents\Downloads\AI-voice-cloning-and-impersonation-detection-features-main\SSL_Anti-spoofing-main\database\ASVspoof_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
out = "colab_subset"
out_audio = os.path.join(out, "flac")
os.makedirs(out_audio, exist_ok=True)
available = {
    os.path.splitext(f)[0]
    for f in os.listdir(audio_dir)
    if f.lower().endswith(".flac")
}
groups = {"bonafide": [], "spoof": []}
with open(protocol, "r") as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 5:
            file_id = parts[1]
            label = parts[-1]
            if label in groups and file_id in available:
                groups[label].append((file_id, line))
random.seed(42)
selected = []
for label in ("bonafide", "spoof"):
    items = groups[label]
    print(label, "available:", len(items))
    selected.extend(random.sample(items, 200))
with open(os.path.join(out, "subset_protocol.txt"), "w") as f:
    for file_id, line in selected:
        shutil.copy2(
            os.path.join(audio_dir, file_id + ".flac"),
            os.path.join(out_audio, file_id + ".flac")
        )
        f.write(line)
print("\nSubset created successfully!")
print("Bonafide: 200")
print("Spoof: 200")
print("Total: 400")
print("Location:", os.path.abspath(out))
