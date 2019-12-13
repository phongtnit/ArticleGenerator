import feedparser
import text_generation
from time import sleep


def main():
    RSS_URL = 'https://www.ft.com/?format=rss'
    feed = feedparser.parse(RSS_URL)
    latest_generation = ''
    model, tokenizer = text_generation.model_tokenizer_initializer('distilgpt2')

    while True:
        latest_title = feed['items'][0]['title']
        latest_title_time = feed['items'][0]['published']
        if latest_generation != latest_title_time:
            text = text_generation.generate_text(model, tokenizer, 200, prompt=latest_title)
            print(latest_title + text)
            latest_generation = latest_title_time
        sleep(60)


if __name__ == '__main__':
    main()
