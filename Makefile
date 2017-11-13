*.eps : mandigostoba-morgaine_hw03.01.py
	python $< 

main.pdf : main.tex *.eps
	pdflatex main.tex

build : main.pdf
