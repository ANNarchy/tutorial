# ANNarchy tutorial

This tutorial presents the neuro-simulator ANNarchy (Artificial Neural Networks architect):

<https://github.com/ANNarchy/ANNarchy>

The webpage and the accompnying slides and notebooks are available at: 

<https://annarchy.github.io/tutorial>.


## Generating the tutorial

The tutorial is made with Quarto <https://quarto.org>.

To generate the webpage in `docs`, use:

```bash
quarto render tutorial
```

To generate the slides:

```bash
quarto render tutorial/slides --to revealjs
```

### Shortcuts for thr slides

* `f`: goes fullscreen.
* `e`: switches to printing mode (`?print-pdf` in the location bar). 
* `s`: opens presenter's view.
* `v`: goes black (pause).
* `m`: opens the menu to select slides.
* `c`: changes the cursor to a pen.
* `b`: opens a chalkboard.
* `CAPSLOCK`: changes the cursor to a laser pointer.
