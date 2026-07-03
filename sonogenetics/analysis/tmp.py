import plotly.graph_objs as go
import plotly.io as pio
from pathlib import Path
from html2image import Html2Image

fig = go.Figure()
test_name = Path(r'C:/sono_data/test.png')

# 1. Save the figure as a temporary HTML string (very fast, no Kaleido required)
html_content = pio.to_html(fig, include_plotlyjs='cdn', full_html=True)

# 2. Initialize the HTML-to-Image converter
hti = Html2Image(output_path=str(test_name.parent))

# 3. Save as a PNG cleanly using your system browser backend
hti.screenshot(
    html_str=html_content,
    save_as=test_name.name,
    size=(700 * 5, 500 * 5)  # Multiplied by 5 to match your scale=5 requirement
)

print("Saved perfectly without Kaleido!")