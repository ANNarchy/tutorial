input_file=tutorial/slides/presentation.qmd

all: html

html:
	quarto render tutorial --to html
	quarto render tutorial/slides --to revealjs

slides: $(input_file)
	quarto render tutorial/slides --to revealjs
