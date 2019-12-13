import spacy
import feedparser
import bs4
import urllib.request
import urllib.error
import picture_scraping
import text_generation
from numpy import random


def find_nth(text, substring, n):
    number_of_iterations = 0
    index_of_substring = 0
    while number_of_iterations < n:
        index_of_substring = text.find(substring, index_of_substring)
        number_of_iterations += 1
    return index_of_substring


def main ():
    RSS_URL = 'https://www.ft.com/?format=rss'
    text = feedparser.parse(RSS_URL)['items'][5]['title']
    model, tokenizer = text_generation.model_tokenizer_initializer('distilgpt2')
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)

    search_query = ''
    for ent in doc.ents:
        search_query += ent.text
        search_query += ' '
    url = picture_scraping.generate_search_url(search_query, tpe='isch')
    list_of_links = picture_scraping.get_links(url)
    img_links = picture_scraping.un_googlify_img_links(list_of_links)
    random_index = random.random_integers(0, len(img_links))
    image = img_links[random_index]

    try:
        urllib.request.urlretrieve(image, f'./Articles/file01')
    except urllib.error.HTTPError:
        print('Forbidden img link, try again!')

    with open('./Articles/Article Template.htm') as article:
        txt = article.read()
        soup = bs4.BeautifulSoup(txt, features='lxml')

    title = soup.find('h2', 'content-subhead')
    title.string = text

    new_img = soup.new_tag('img', class_='pure-img-responsive', src='./file01')
    soup.body.a.append(new_img)

    article = text_generation.generate_text(model, tokenizer, 250, prompt=text, temperature=1)
    slice_index = find_nth(article, '.', 4)
    first_part = article[0:slice_index+1]
    second_part = article[slice_index+1:]

    first_paragraph = soup.find('p')
    first_paragraph.string = first_part
    second_paragraph = first_paragraph.find_next('p')
    second_paragraph.string = second_part

    with open('./Articles/generated_article.htm', 'w') as edited_file:
        edited_file.write(str(soup))


if __name__ == '__main__':
    main()
