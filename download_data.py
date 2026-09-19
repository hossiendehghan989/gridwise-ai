from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip"
root = Path(__file__).parent
zip_path = root / "data" / "uci-energy.zip"
out = root / "data"
out.mkdir(exist_ok=True)
print("Downloading UCI Appliances Energy Prediction dataset...")
with urlopen(URL) as response:
    zip_path.write_bytes(response.read())
with ZipFile(zip_path) as zf:
    zf.extractall(out)
zip_path.unlink()
print("Dataset ready in", out)
