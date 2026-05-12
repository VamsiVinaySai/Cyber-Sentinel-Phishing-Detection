import re
import math
from urllib.parse import urlparse
from difflib import SequenceMatcher

# Top legitimate domains for similarity comparison
TOP_DOMAINS = [
    'google.com', 'youtube.com', 'facebook.com', 'twitter.com', 'instagram.com',
    'linkedin.com', 'wikipedia.org', 'amazon.com', 'yahoo.com', 'reddit.com',
    'netflix.com', 'microsoft.com', 'apple.com', 'github.com', 'stackoverflow.com',
    'whatsapp.com', 'telegram.org', 'discord.com', 'twitch.tv', 'tiktok.com',
    'paypal.com', 'ebay.com', 'walmart.com', 'bankofamerica.com', 'chase.com',
    'wellsfargo.com', 'citibank.com', 'hdfc.com', 'sbi.co.in', 'icicibank.com',
    'axisbank.com', 'paytm.com', 'phonepe.com', 'flipkart.com', 'myntra.com',
    'zomato.com', 'swiggy.com', 'uber.com', 'airbnb.com', 'booking.com',
    'makemytrip.com', 'irctc.co.in', 'uidai.gov.in', 'zoom.us', 'dropbox.com',
    'gmail.com', 'outlook.com', 'adobe.com', 'salesforce.com', 'wordpress.com',
    'shopify.com', 'godaddy.com', 'cloudflare.com', 'spotify.com', 'pinterest.com',
    'quora.com', 'medium.com', 'coursera.org', 'udemy.com', 'khanacademy.org',
    'bbc.com', 'cnn.com', 'nytimes.com', 'ndtv.com', 'timesofindia.com',
    'thehindu.com', 'cricbuzz.com', 'espn.com', 'hotstar.com', 'primevideo.com',
    'kaggle.com', 'huggingface.co', 'pytorch.org', 'tensorflow.org', 'openai.com',
    'anthropic.com', 'notion.so', 'figma.com', 'canva.com', 'trello.com',
    'slack.com', 'atlassian.com', 'jira.com', 'confluence.com', 'bitbucket.org',
    'gitlab.com', 'npmjs.com', 'pypi.org', 'docker.com', 'kubernetes.io',
    'digitalocean.com', 'heroku.com', 'vercel.com', 'netlify.com', 'firebase.google.com',
]

TLD_LEGIT_PROB = {
    'com': 0.5229, 'org': 0.0799, 'net': 0.0522, 'edu': 0.0100,
    'gov': 0.0050, 'uk': 0.0281, 'de': 0.0326, 'in': 0.0050,
    'ru': 0.0180, 'jp': 0.0231, 'fr': 0.0150, 'au': 0.0120,
    'co': 0.0070, 'info': 0.0060, 'biz': 0.0040, 'io': 0.0080,
    'us': 0.0060, 'ca': 0.0110, 'br': 0.0100, 'cn': 0.0090,
}

def _entropy(s):
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())

def _url_similarity_index(hostname):
    """
    100  = exact match to a known legitimate domain (or its subdomain)
    <100 = typo / fake domain — lower score = more suspicious
    """
    hostname = hostname.lower()
    if hostname.startswith('www.'):
        hostname = hostname[4:]

    # Exact match
    if hostname in TOP_DOMAINS:
        return 100.0

    # Legitimate subdomain  e.g. mail.google.com, docs.google.com
    for domain in TOP_DOMAINS:
        if hostname.endswith('.' + domain):
            return 100.0

    # Best fuzzy similarity — gogle.com → ~91, not 100
    best = max(SequenceMatcher(None, hostname, d).ratio() for d in TOP_DOMAINS)
    return round(best * 100, 4)

def _char_continuation_rate(url, letter_count):
    if not letter_count:
        return 0.0
    max_run = cur = 0
    prev_type = None
    for c in url:
        t = 'a' if c.isalpha() else ('d' if c.isdigit() else 'o')
        cur = cur + 1 if t == prev_type else 1
        max_run = max(max_run, cur)
        prev_type = t
    return max_run / letter_count

# MUST match CSV_FEATURE_COLS order in train_model.py exactly
FEATURE_NAMES = [
    'URLLength', 'DomainLength', 'IsDomainIP', 'TLDLength',
    'URLSimilarityIndex', 'CharContinuationRate', 'TLDLegitimateProb',
    'URLCharProb', 'NoOfSubDomain', 'HasObfuscation',
    'NoOfObfuscatedChar', 'ObfuscationRatio', 'NoOfLettersInURL',
    'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL',
    'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL',
    'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL', 'IsHTTPS',
]

def extract_features(url):
    """
    Returns 22 pure URL-lexical features — NO HTML scraping.
    Order matches FEATURE_NAMES / CSV_FEATURE_COLS in train_model.py.
    """
    if not url or not isinstance(url, str):
        return [0] * len(FEATURE_NAMES)

    if not re.match(r'^https?://', url):
        url = 'http://' + url

    try:
        parsed = urlparse(url)
    except Exception:
        return [0] * len(FEATURE_NAMES)

    scheme   = parsed.scheme or ''
    hostname = parsed.netloc or ''
    path     = parsed.path or ''
    query    = parsed.query or ''

    host_no_port = hostname.split(':')[0].lower()
    parts  = host_no_port.split('.')
    tld    = parts[-1] if len(parts) > 1 else ''
    domain = parts[-2] if len(parts) > 1 else host_no_port

    url_len      = len(url)
    domain_len   = len(domain)
    is_ip        = 1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', host_no_port) else 0
    tld_len      = len(tld)

    url_sim_index  = _url_similarity_index(host_no_port)   # THE key feature

    letter_count   = sum(c.isalpha() for c in url)
    digit_count    = sum(c.isdigit() for c in url)

    char_cont_rate = _char_continuation_rate(url, letter_count)
    tld_legit_prob = TLD_LEGIT_PROB.get(tld, 0.001)
    url_char_prob  = _entropy(url) / 8.0          # normalized to ~0-1

    num_subdomains = max(0, len(parts) - 2)

    has_obfusc   = 1 if ('%' in url or '0x' in url.lower()) else 0
    num_obfusc   = url.lower().count('0x') + url.count('%2') + url.count('%3')
    obfusc_ratio = num_obfusc / url_len if url_len else 0

    letter_ratio = letter_count / url_len if url_len else 0
    digit_ratio  = digit_count  / url_len if url_len else 0

    equals_count = url.count('=')
    qmark_count  = url.count('?')
    amp_count    = url.count('&')
    other_special = sum(
        1 for c in url if not c.isalnum() and c not in './:?=&@#-_~%+'
    )
    special_ratio = other_special / url_len if url_len else 0

    is_https = 1 if scheme == 'https' else 0

    return [
        url_len,        # URLLength
        domain_len,     # DomainLength
        is_ip,          # IsDomainIP
        tld_len,        # TLDLength
        url_sim_index,  # URLSimilarityIndex
        char_cont_rate, # CharContinuationRate
        tld_legit_prob, # TLDLegitimateProb
        url_char_prob,  # URLCharProb
        num_subdomains, # NoOfSubDomain
        has_obfusc,     # HasObfuscation
        num_obfusc,     # NoOfObfuscatedChar
        obfusc_ratio,   # ObfuscationRatio
        letter_count,   # NoOfLettersInURL
        letter_ratio,   # LetterRatioInURL
        digit_count,    # NoOfDegitsInURL
        digit_ratio,    # DegitRatioInURL
        equals_count,   # NoOfEqualsInURL
        qmark_count,    # NoOfQMarkInURL
        amp_count,      # NoOfAmpersandInURL
        other_special,  # NoOfOtherSpecialCharsInURL
        special_ratio,  # SpacialCharRatioInURL
        is_https,       # IsHTTPS
    ]



# import re
# import math
# from urllib.parse import urlparse
# from difflib import SequenceMatcher

# # Top legitimate domains for similarity comparison
# TOP_DOMAINS = [
#     'google.com', 'youtube.com', 'facebook.com', 'twitter.com', 'instagram.com',
#     'linkedin.com', 'wikipedia.org', 'amazon.com', 'yahoo.com', 'reddit.com',
#     'netflix.com', 'microsoft.com', 'apple.com', 'github.com', 'stackoverflow.com',
#     'whatsapp.com', 'telegram.org', 'discord.com', 'twitch.tv', 'tiktok.com',
#     'paypal.com', 'ebay.com', 'walmart.com', 'bankofamerica.com', 'chase.com',
#     'wellsfargo.com', 'citibank.com', 'hdfc.com', 'sbi.co.in', 'icicibank.com',
#     'axisbank.com', 'paytm.com', 'phonepe.com', 'flipkart.com', 'myntra.com',
#     'zomato.com', 'swiggy.com', 'uber.com', 'airbnb.com', 'booking.com',
#     'makemytrip.com', 'irctc.co.in', 'uidai.gov.in', 'zoom.us', 'dropbox.com',
#     'gmail.com', 'outlook.com', 'adobe.com', 'salesforce.com', 'wordpress.com',
#     'shopify.com', 'godaddy.com', 'cloudflare.com', 'spotify.com', 'pinterest.com',
#     'quora.com', 'medium.com', 'coursera.org', 'udemy.com', 'khanacademy.org',
#     'bbc.com', 'cnn.com', 'nytimes.com', 'ndtv.com', 'timesofindia.com',
#     'thehindu.com', 'cricbuzz.com', 'espn.com', 'hotstar.com', 'primevideo.com',
# ]

# TLD_LEGIT_PROB = {
#     'com': 0.5229, 'org': 0.0799, 'net': 0.0522, 'edu': 0.0100,
#     'gov': 0.0050, 'uk': 0.0281, 'de': 0.0326, 'in': 0.0050,
#     'ru': 0.0180, 'jp': 0.0231, 'fr': 0.0150, 'au': 0.0120,
#     'co': 0.0070, 'info': 0.0060, 'biz': 0.0040, 'io': 0.0080,
# }

# SUSPICIOUS_KEYWORDS = [
#     'login', 'secure', 'account', 'update', 'banking', 'confirm', 'verify',
#     'signin', 'free', 'lucky', 'bonus', 'prize', 'winner', 'password',
#     'credential', 'wallet', 'crypto', 'urgent', 'alert', 'suspended', 'limited',
# ]

# def _entropy(s):
#     if not s:
#         return 0.0
#     freq = {}
#     for c in s:
#         freq[c] = freq.get(c, 0) + 1
#     n = len(s)
#     return -sum((v / n) * math.log2(v / n) for v in freq.values())

# def _url_similarity_index(hostname):
#     """
#     THE KEY FIX:
#     - Exact match to known legit domain  → 100
#     - Legitimate subdomain of known domain → 100
#     - Typo/fake (e.g. gogle.com, g00gle.com) → 50-95 (NOT 100)
#     - Completely unrelated domain → low score
#     The model learned that score < 100 with high similarity = phishing pattern.
#     """
#     hostname = hostname.lower().replace('www.', '')

#     # Exact match
#     if hostname in TOP_DOMAINS:
#         return 100.0

#     # Legitimate subdomain (e.g. mail.google.com)
#     for domain in TOP_DOMAINS:
#         if hostname.endswith('.' + domain):
#             return 100.0

#     # Find best similarity to any known domain
#     best = 0.0
#     for domain in TOP_DOMAINS:
#         r = SequenceMatcher(None, hostname, domain).ratio()
#         if r > best:
#             best = r

#     return round(best * 100, 4)

# def _char_continuation_rate(url, letter_count):
#     if not letter_count:
#         return 0.0
#     max_run = cur = 0
#     prev_type = None
#     for c in url:
#         t = 'a' if c.isalpha() else ('d' if c.isdigit() else 'o')
#         cur = cur + 1 if t == prev_type else 1
#         max_run = max(max_run, cur)
#         prev_type = t
#     return max_run / letter_count

# def extract_features(url):
#     """
#     Returns 29 features in the EXACT same order as CSV_FEATURE_COLS in train_model.py.
#     """
#     if not url or not isinstance(url, str):
#         return [0] * 29

#     if not re.match(r'^https?://', url):
#         url = 'http://' + url

#     try:
#         parsed = urlparse(url)
#     except Exception:
#         return [0] * 29

#     scheme   = parsed.scheme or ''
#     hostname = parsed.netloc or ''
#     path     = parsed.path or ''
#     query    = parsed.query or ''

#     host_no_port = hostname.split(':')[0].lower()
#     parts  = host_no_port.split('.')
#     tld    = parts[-1] if len(parts) > 1 else ''
#     domain = parts[-2] if len(parts) > 1 else host_no_port

#     url_len      = len(url)
#     domain_len   = len(domain)
#     is_ip        = 1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', host_no_port) else 0
#     tld_len      = len(tld)

#     # ↓ THE CRITICAL FIX
#     url_sim_index   = _url_similarity_index(host_no_port)

#     letter_count    = sum(c.isalpha() for c in url)
#     char_cont_rate  = _char_continuation_rate(url, letter_count)
#     tld_legit_prob  = TLD_LEGIT_PROB.get(tld, 0.001)
#     url_char_prob   = _entropy(url) / 8.0  # normalized entropy

#     num_subdomains  = max(0, len(parts) - 2)
#     has_obfusc      = 1 if ('%' in url or '0x' in url.lower()) else 0
#     num_obfusc      = url.lower().count('0x') + url.count('%2') + url.count('%3')
#     obfusc_ratio    = num_obfusc / url_len if url_len else 0

#     digit_count     = sum(c.isdigit() for c in url)
#     letter_ratio    = letter_count / url_len if url_len else 0
#     digit_ratio     = digit_count / url_len if url_len else 0

#     equals_count    = url.count('=')
#     qmark_count     = url.count('?')
#     amp_count       = url.count('&')
#     other_special   = sum(1 for c in url if not c.isalnum() and c not in './:?=&@#-_~%+')
#     special_ratio   = other_special / url_len if url_len else 0
#     is_https        = 1 if scheme == 'https' else 0

#     path_words          = re.split(r'[\W_]+', path)
#     longest_path_token  = max((len(w) for w in path_words if w), default=0)

#     url_lower           = url.lower()
#     suspicious_count    = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)
#     title_match_score   = max(0.0, 1.0 - suspicious_count * 0.2)

#     has_social          = 1 if any(s in url_lower for s in ['facebook', 'twitter', 'instagram', 'linkedin']) else 0
#     domain_title_score  = url_sim_index / 100.0
#     url_domain_ratio    = domain_len / url_len if url_len else 0

#     char_counts         = {}
#     for c in url:
#         char_counts[c] = char_counts.get(c, 0) + 1
#     char_repeat_rate    = max(char_counts.values()) / url_len if url_len else 0

#     slash_count = url.count('/')

#     # ORDER MUST MATCH CSV_FEATURE_COLS in train_model.py exactly
#     return [
#         url_len,            # URLLength
#         domain_len,         # DomainLength
#         is_ip,              # IsDomainIP
#         tld_len,            # TLDLength
#         url_sim_index,      # URLSimilarityIndex  ← KEY
#         char_cont_rate,     # CharContinuationRate
#         tld_legit_prob,     # TLDLegitimateProb
#         url_char_prob,      # URLCharProb
#         num_subdomains,     # NoOfSubDomain
#         has_obfusc,         # HasObfuscation
#         num_obfusc,         # NoOfObfuscatedChar
#         obfusc_ratio,       # ObfuscationRatio
#         letter_count,       # NoOfLettersInURL
#         letter_ratio,       # LetterRatioInURL
#         digit_count,        # NoOfDegitsInURL
#         digit_ratio,        # DegitRatioInURL
#         equals_count,       # NoOfEquals
#         qmark_count,        # NoOfQMarkInURL
#         amp_count,          # NoOfAmpersandInURL
#         other_special,      # NoOfOtherSpecialCharsInURL
#         special_ratio,      # SpacialCharRatioInURL
#         is_https,           # IsHTTPS
#         longest_path_token, # LongestPathTokenLength
#         title_match_score,  # URLTitleMatchScore
#         has_social,         # HasSocialNet
#         domain_title_score, # DomainTitleMatchScore
#         url_domain_ratio,   # URLDomainRatio
#         char_repeat_rate,   # CharRepeatRate
#         slash_count,        # NoOfSlashInURL
#     ]
