#! /bin/bash
for f in *.csv 
  do 
    gnuplot -e "filename='${f%.*}'" .plot 
    # rm ${f%.*}.eps
  done
for f in *.eps 
  do 
  convert -density 600 -flatten ${f%.*}.eps ${f%.*}.png
done
    