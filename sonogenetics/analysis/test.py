import plotly.graph_objects as go
from playwright.sync_api import sync_playwright


def save_plotly_png(fig, filename="output.png", width=1000, height=600):
    html_content = fig.to_html(include_plotlyjs="cdn", full_html=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": height})
        page.set_content(html_content)
        # Wait for Plotly chart to load completely
        page.wait_for_selector(".main-svg")
        page.screenshot(path=filename)
        browser.close()


# Example usage
fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
save_plotly_png(fig, "test_playwright.png")
print("Saved PNG without Kaleido!")