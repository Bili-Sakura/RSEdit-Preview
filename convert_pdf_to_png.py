"""Convert disaster figure PDFs to PNG format."""
from pathlib import Path
from pdf2image import convert_from_path

IMAGES_DIR = Path("docs/static/images")
PDFS = [
    "figures-big-big-Guatemala-Volcano.pdf",
    "figures-big-big-Hurricane-Florence.pdf",
    "figures-big-big-Joplin-Tornado.pdf",
    "figures-big-big-Mexico-Earthquake.pdf",
    "figures-big-big-Palu-Tsunami.pdf",
    "figures-big-big-Portugal-Wildfire.pdf",
]

for pdf_name in PDFS:
    pdf_path = IMAGES_DIR / pdf_name
    if not pdf_path.exists():
        print(f"Skip (not found): {pdf_name}")
        continue
    png_name = pdf_name.replace(".pdf", ".png")
    png_path = IMAGES_DIR / png_name
    print(f"Converting {pdf_name} -> {png_name} ...", end=" ")
    try:
        images = convert_from_path(str(pdf_path), dpi=150)
        images[0].save(str(png_path), "PNG")
        print("OK")
    except Exception as e:
        print(f"Error: {e}")

print("Done.")
