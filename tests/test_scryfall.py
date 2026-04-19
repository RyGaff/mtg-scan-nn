from unittest.mock import patch, MagicMock
from src.scryfall import get_unique_artwork_url, download_unique_artwork


def test_get_unique_artwork_url_filters_for_unique_artwork():
    bulk_response = {
        "data": [
            {"type": "default_cards", "download_uri": "https://x/default.json"},
            {"type": "unique_artwork", "download_uri": "https://x/unique.json"},
        ]
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = bulk_response
    mock_resp.raise_for_status = MagicMock()
    with patch("src.scryfall.requests.get", return_value=mock_resp):
        url = get_unique_artwork_url()
    assert url == "https://x/unique.json"


def test_download_unique_artwork_writes_json(tmp_path):
    cards = [{"id": "abc", "name": "Lightning Bolt"}]
    mock_bulk = MagicMock()
    mock_bulk.json.return_value = {"data": [
        {"type": "unique_artwork", "download_uri": "https://x/u.json"}]}
    mock_bulk.raise_for_status = MagicMock()
    mock_data = MagicMock()
    mock_data.json.return_value = cards
    mock_data.raise_for_status = MagicMock()
    dest = tmp_path / "u.json"
    with patch("src.scryfall.requests.get", side_effect=[mock_bulk, mock_data]):
        download_unique_artwork(dest)
    import json
    assert json.loads(dest.read_text()) == cards
