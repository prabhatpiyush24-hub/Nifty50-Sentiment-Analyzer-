import re

with open("d:/NLP_Project/data_fetcher.py", "r", encoding="utf-8") as f:
    content = f.read()

# Change typing imports if needed (Dict is already imported)
content = content.replace("def _fetch_rss_headlines(ticker: str, company_name: str, n: int = 10) -> List[str]:", "def _fetch_rss_headlines(ticker: str, company_name: str, n: int = 10) -> List[Dict[str, str]]:")

# Update yahoo RSS append
content = content.replace('headlines.append(title)', 'headlines.append({"headline": title, "published": entry.get("published", entry.get("pubDate", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))})', 1)

# Update Google news RSS append
content = content.replace('headlines.append(title)', 'headlines.append({"headline": title, "published": entry.get("published", entry.get("pubDate", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))})', 1)

# Now fix the Google news condition `title not in headlines`
content = content.replace('if title and len(title) > 15 and title not in headlines:', 'if title and len(title) > 15 and not any(h.get("headline") == title for h in headlines):')

# Update mock headlines signature
content = content.replace("def _generate_mock_headlines(ticker: str, company_name: str, n: int = 10) -> List[str]:", "def _generate_mock_headlines(ticker: str, company_name: str, n: int = 10) -> List[Dict[str, str]]:")

# Update mock headline appends
content = content.replace('headlines.append(headline)', '''
        random_minutes = random.randint(1, 1440)
        pub_date = (datetime.datetime.now() - datetime.timedelta(minutes=random_minutes)).strftime("%Y-%m-%d %H:%M:%S")
        headlines.append({"headline": headline, "published": pub_date})''')

# Update fetch_headlines signature
content = content.replace("def fetch_headlines(ticker: str, company_name: str, n: int = MAX_HEADLINES_PER_TICKER) -> List[str]:", "def fetch_headlines(ticker: str, company_name: str, n: int = MAX_HEADLINES_PER_TICKER) -> List[Dict[str, str]]:")

# Update fetch_all_headlines signature
content = content.replace("def fetch_all_headlines(tickers_sectors: Optional[Dict[str, str]] = None) -> Dict[str, List[str]]:", "def fetch_all_headlines(tickers_sectors: Optional[Dict[str, str]] = None) -> Dict[str, List[Dict[str, str]]]:")

with open("d:/NLP_Project/data_fetcher.py", "w", encoding="utf-8") as f:
    f.write(content)
