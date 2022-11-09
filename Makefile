input_file=slides/presentation.qmd

all: html

html: $(input_file)
	quarto render webpage --to html
	quarto render slides --to revealjs

slides: $(input_file)
	quarto render slides --to revealjs
