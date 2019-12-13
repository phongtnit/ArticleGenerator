from requests_html import HTMLSession


def generate_search_url(query, tld='com', lang='en', tbs='0', safe='off', tpe='', country=''):
    """Generates the url for a search on google."""
    url_search = f"https://www.google.{tld}/search?hl={lang}&q={query}&" \
                 f"btnG=Google+Search&tbs={tbs}&safe={safe}&tbm={tpe}&" \
                 f"cr={country}"
    return url_search


def render_url(url):
    """starts a HTMLSession and renders the page."""
    session = HTMLSession()
    r = session.get(url)
    return r


def replace_iteration(string, dict_of_replacements):
    """Calls the replace method multiple times on a string.
    Does the replacements based on the keys and values in the input dict.
    """
    for i, j in dict_of_replacements.items():
        string = string.replace(i, j)
    return string


def img_search_string_shortening(google_link):
    """Shortens the google link that requests-html found to only include the part containing the image url."""
    start_index = google_link.find('https')
    end_index = google_link.find('&imgrefurl')
    only_img_link = google_link[start_index:end_index]
    return only_img_link


def get_links(url):
    """
    Gets all the links available on a web-page.
    :param url: Any url.
    :return: A list of links.
    """
    r = render_url(url)
    list_of_links = list(r.html.links)
    return list_of_links


def un_googlify_img_links(list_of_links):
    """Removes all the google search stuff around the google links that requests-html found.
    Returns a list of completely normal https links of the image search result.
    """
    dict_of_replacements = {'%3A': ':', '%2F': '/', '%3F': '?', '%3D': '=', '%26': '&'}
    list_of_img_links = [replace_iteration(img_search_string_shortening(link), dict_of_replacements) for link in
                         list_of_links if '/imgres?imgurl=https' in link]
    return list_of_img_links


def main():
    query = 'cats'
    params = dict()
    params['tpe'] = 'isch'
    ulr = generate_search_url(query, **params)
    print(un_googlify_img_links(ulr))


if __name__ == '__main__':
    main()
