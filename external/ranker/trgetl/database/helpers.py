import re
import time
import urllib


def csvit(it, sep=",", quote=""):
    """Convert iterable to sep-joined string of __str__ representations"""
    res = sep.join(strit(it, quote=quote))
    return res


def query_urlenc(s):
    """Urlencode query=query_text parameter for CH."""
    _s = s.replace("\n", " ").replace("\t", " ")
    while "  " in _s:
        _s = _s.replace("  ", " ")
    return urllib.parse.urlencode({"query": _s})


def clean_html(raw_html):
    cleanr = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    cleantext = re.sub(cleanr, "", raw_html)
    cleantext = cleantext.replace("\r", " ").replace("\n", " ")
    cleantext = cleantext.replace("  ", " ").replace("  ", " ")
    return str(cleantext.strip())


def timeit(tm):
    """Substract fixed start time from current time"""
    return time.time() - tm


def strit(it, quote=""):
    """Convert iterable to iterable of __str__ representations"""
    return [quote + str(e) + quote for e in it]
