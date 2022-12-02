input_file=src/slides/presentation.qmd

all: html

html:
	quarto render src --to html
	quarto render src/slides --to revealjs

slides: $(input_file)
	quarto render src/slides --to revealjs
