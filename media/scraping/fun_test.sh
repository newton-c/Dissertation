#!/bin/bash -w

# directories need to be made in the WD before running

# This first function scrapes the URLS fro the relevant section
function getURLS { 
        for i in {1..$maxpages}
        do
                curl "$url$i" |
                grep "$link_name" |
                sed "s/$pre_link_find/$pre_link_replace/" |
                sed "s/$post_link_find/$post_link_replace/" |
                uniq 
        done > $file_name

        }

#getURLS 

# This function takes the URLS from the first a get the article text
function getArticles {
        counter=1
        for i in $(cat $file_name)
        do
                curl $i |
                html2text |
                sed -n "$cut_head" |
                sed -n "$cut_tail" |
                tail -n+$cut_lines > $article_file
                counter=$((counter+1))
        done
}

#getArticles

#### Scraping ####

## Bogota ##

# El Espectador: www.elespectador.com
maxpages=2
url=https://www.elespectador.com/tags/farc/
link_name="<a class=\"Card-FullArticle\" href=.*"
pre_link_find="<a class=\"Card-FullArticle\" href="
pre_link_replace="https\:\/\/www\.elespectador\.com/"
post_link_find="\">Ver noticia completa.*"
post_link_replace=
file_name=test_urls.txt

cut_head='/Escuchar este artículo/,$p'
cut_tail='/Conecta_con_la_verdad._Suscríbete_a_elpais.com.co/q;p'
cut_lines=2
article_file=

getURLS
#getArticles

## El País
#
#maxpages=2
#url=https://www.elpais.com.co/noticias/farc
#link_name="href=.*class=\"news-title\""
#pre_link_find="^.*href=\""
#pre_link_replace="https\:\/\/www\.elpais\.com\.co"
#post_link_find="\"\stitle.*"
#post_link_replace=
#
#
#file_name="test_urls.txt"
#cut_head='/Escuchar este artículo/,$p'
#cut_tail='/Conecta_con_la_verdad._Suscríbete_a_elpais.com.co/q;p'
#cut_lines=2

#getURLS
#getArticles
