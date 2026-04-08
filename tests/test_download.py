from unittest.mock import MagicMock, patch

from scripts.download import fetch_script_text, fetch_script_urls

SAMPLE_INDEX = """<html><body>
<a href="TheStakeout.html">The Stakeout</a>
<a href="TheRobbery.htm">The Robbery</a>
<a href="https://external.com/other">External</a>
<a href="seinfeld-scripts.html">Home</a>
</body></html>"""

SAMPLE_SCRIPT = """<html><body>
<p>[JERRY'S APARTMENT]</p>
<p>JERRY: Hello.</p>
<p>GEORGE: Hi there.</p>
</body></html>"""


def test_fetch_script_urls_returns_episode_links():
    resp = MagicMock()
    resp.text = SAMPLE_INDEX
    with patch("scripts.download.requests.get", return_value=resp):
        urls = fetch_script_urls()
    assert "http://seinfeldscripts.com/TheStakeout.html" in urls
    assert "http://seinfeldscripts.com/TheRobbery.htm" in urls


def test_fetch_script_urls_excludes_external_and_index():
    resp = MagicMock()
    resp.text = SAMPLE_INDEX
    with patch("scripts.download.requests.get", return_value=resp):
        urls = fetch_script_urls()
    assert not any("external.com" in u for u in urls)
    assert not any("seinfeld-scripts.html" in u for u in urls)


def test_fetch_script_text_strips_html():
    resp = MagicMock()
    resp.text = SAMPLE_SCRIPT
    with patch("scripts.download.requests.get", return_value=resp):
        text = fetch_script_text("http://seinfeldscripts.com/TheStakeout.html")
    assert "JERRY" in text
    assert "<p>" not in text
