(TeX-add-style-hook
 "weave-test"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "a4paper" "12pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("algpseudocode" "noend")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art12"
    "mathtools"
    "amsfonts"
    "amssymb"
    "amsmath"
    "bm"
    "commath"
    "multicol"
    "algorithmicx"
    "tkz-graph"
    "algorithm"
    "fancyhdr"
    "minted"
    "pgfplots"
    "textgreek"
    "graphicx"
    "fancyvrb"
    "hyperref"
    "algpseudocode"))
 :latex)

