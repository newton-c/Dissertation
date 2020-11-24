#!/bin/bash -w

# directories need to be made in the WD before running

# This first function scrapes the URLS for the relevant section
function getURLS { 
                wget "$url" |
                grep "$link_name" |
                sed "s/$pre_link_find/$pre_link_replace/" |
                sed "s/$post_link_find/$post_link_replace/" |
                uniq > $file_name

        }

#getURLS 

# This function takes the URLS from the first a get the article text
function getArticles {
        counter=1
        for i in $(cat $file_name)
        do
                wget $i |
                html2text |
                sed -n "$cut_head" |
                sed -n "$cut_tail" |
                tail -n+$cut_lines > $article_file$counter.txt
                counter=$((counter+1))
        done
}

#### variables for getURLS command

# url should include the page range in square brackets []
# for example if you want to scrape all results from a search with 20 pages:
# https://www.exampleurl.com/results/?page=[1-20]
url=

# the HTML that identifies the links to the articles you want to scrape
link_name=""

# what occurs before the link to be deleted, often ^.*href=\""
# this selects all characters from the beginning of the line to the link
pre_link_find=""

# what to replace before the link, usually the front of the URL
pre_link_replace=""

# what occurs after the link to be deleted
post_link_find=""

# what to replace after the link, often blank
post_link_replace=

# name of file to store the list of urls, should be .txt format
file_name=


#### variables for getArticles command

# a pattern to delete everything before it to the beginning of the page
cut_head='//,$p'

# a pattern to delete everything after it to the end of the page
cut_tail='//q;p'

# if cut_tail leaves some unwanted line, this it the number of
#lines from the bottom to delete
cut_lines=

# should specify directory and a name for files (article_file=director/file_)
article_file=

getURLS
getArticles
