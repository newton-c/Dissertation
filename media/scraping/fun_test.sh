#!/bin/bash -w

maxpages=2
url=https://www.elpais.com.co/noticias/farc
link_name="href=.*class=\"news-title\""
pre_link_find="^.*href=\""
pre_link_replace="https\:\/\/www\.elpais\.com\.co"
post_link_find="\"\stitle.*"
post_link_replace=

function getURLS { 
        for i in {1..2}
        do
                curl $url |
                grep "$link_name" |
                sed "s/$pre_link_find/$pre_link_replace/" |
                sed "s/$post_link_find/$post_link_replace/" |
                uniq 
        done > test_urls.txt

        }

getURLS 

file_name="elpais/elpais_urls.txt"
cut_head='Escuchar este artículo/,$p'
cut_tail='Conecta_con_la_verdad._Suscríbete_a_elpais.com.co/q;p'

function getArticles {
        for $i in $(cat $file_name)
        do
                curl $i |
                html2text |
                sed -n "$cut_head" |
                sed -n "$cut_tail" |
                tail -n+$cut_lines > test_text.txt
                counter=$((counter+1))
        done
}












