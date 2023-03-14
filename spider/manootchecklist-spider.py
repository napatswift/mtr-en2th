import scrapy

class BlogSpider(scrapy.Spider):
    name = 'blogspider'
    start_urls = ['https://manootchecklist.wordpress.com/2022/11/20/engtothlyrics-westlife-fool-again/']

    def parse(self, response):

        music_title = response.css('h1.post-title::text').get()
        if music_title is not None:
            yield {
                    'title': music_title,
                    'lyrics': response\
                            .css('div.post-content')\
                            .xpath('p//text()').getall()
                }


        for music_page in response\
                .css('h2.post-title')\
                .xpath('a/@href')\
                .getall():
            yield scrapy.Request(
                    response.urljoin(music_page), self.parse)


        next_page = response\
                .css('div.nav-previous')\
                .xpath('a/@href').get()

        next_page_list = [next_page] + response\
                .css('nav.jp-relatedposts-i2')\
                .xpath('div/ul/li/a/@href').getall()
        for next_page in next_page_list:
            yield scrapy.Request(response.urljoin(next_page), self.parse)

