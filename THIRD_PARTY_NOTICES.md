# License scope

The MIT license covers the research code and repository documentation. It does
not relicense the manuscript, bibliography, or publication figures in `paper/`.
Those retain their existing rights. Posting the manuscript on arXiv or OpenReview
does not change this code license.

The bundled `paper/neurips_2026.sty` and `paper/environ.sty` are copied from the
paper's source without modification; their existing headers are preserved.
`environ` is by Will Robertson and states the LaTeX Project Public License in
the file. These style files are outside the repository's MIT grant.
The complete, unmodified [CTAN environ v0.3 archive](paper/vendor/environ/environ.tds.zip)
is bundled alongside its [upstream README](paper/vendor/environ/README).
It includes `environ.dtx`, `environ.ins`, and `environ.pdf`, as named in the
style header, and was obtained from
<https://mirrors.ctan.org/install/macros/latex/contrib/environ.tds.zip>.
The bundled `environ.sty` is byte-identical to that archive's copy.

`paper/trimspaces.sty` is an unmodified copy of `tex/latex/trimspaces/trimspaces.sty`
from the [CTAN trimspaces v1.1 distribution](https://ctan.org/pkg/trimspaces).
The package is by Will Robertson, copyright 2009, and is released under the
[LaTeX Project Public License](https://www.latex-project.org/lppl/).
Its upstream `.sty` does not contain the author/license notice: that notice is
preserved verbatim in [the upstream README](paper/vendor/trimspaces/README).
The [complete, unmodified CTAN archive](paper/vendor/trimspaces/trimspaces.tds.zip)
includes the package's source, installation file, and documentation. It was
obtained from <https://mirrors.ctan.org/install/macros/latex/contrib/trimspaces.tds.zip>.
All bundled trimspaces files are outside the repository's MIT grant.

The 16 benchmark baselines are the authors' local implementations, not code
downloaded from the cited papers. Algorithm citations are in `paper/references.bib`
and `docs/ALGORITHMS.md`. Installed dependencies retain their own licenses.
