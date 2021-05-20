MIN_CITATIONS = 5


# Every heading that contains "background" or "literaure" from within
# the top-2000 section headings (by # of paragraphs, not unique papers).
BackgroundHeadings = {'background', 'related work', 'literature review', 'related works', 'background:',
                      'theoretical background', 'literature search', 'ii. related work', 'related literature',
                      'background and related work', 'literature survey', 'historical background', '| background',
                      'literature', 'background and rationale {6a}', 'research background', 'review of literature',
                      'background information', 'comparison with existing literature', 'abstract background',
                      'background & summary', 'background and rationale'}


def bib_to_key(bib):
    """
    Each different (lowercase title, first author's last name) pair identifies a unique reference paper.
    """
    title, authors, year = bib['title'], bib['authors'], bib['year']

    try:
        first_author = authors[0]['last']
    except:
        first_author = None

    if len(title):
        return (title.lower(), first_author)  # len(authors), year

    return None


def bib_to_citation(bib):
    title, authors, year = bib['title'], bib['authors'], bib['year']

    if len(authors) == 0:
        citation = None
    elif len(authors) == 2:
        citation = ' and '.join([authors[0]['last'], authors[1]['last']])
    else:
        citation = authors[0]['last'] + (" et al." if len(authors) > 1 else "")

    if citation:
        citation += f", {bib['year'] or 'n.d.'}, "
        citation += ' '.join(bib['title'].split()[:12])
        return f"{{{citation}}}"

    return None


def list_of_unique_dicts(dicts):
    """
    Source: https://stackoverflow.com/a/38521207/1493011
    """

    return [dict(s) for s in set(frozenset(d.items()) for d in dicts)]
