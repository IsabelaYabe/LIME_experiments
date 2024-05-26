We collected this dataset on January 23, 2016

We crawled pages from the DMOZ directory, and supplemented the atheism category
from pages listed in the following two websites:
- http://www.atheismunited.com/wiki/Huge_list_of_atheist_agnostic_skeptic_humanist_websites#International
- http://www.atheistsites.net/

We ran each page through the w3m text based browser to render it as text, and kept pages for which the output was larger than 700 characters.
There are 1000 pages in Christianity, and 819 in Atheism. In our experiments, we
balanced the classes out (both with 819).

The id - URL mapping is found in atheism_urls.txt and christianity_urls.txt
