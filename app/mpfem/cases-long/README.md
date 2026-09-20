# Long-running cases (kept, not run by the regression)

Cases of the same kind as `app/mpfem/cases`, whose COMSOL reference costs
minutes and whose mpfem solve costs more. `app/mpfem/tests/python` does not
read this directory — the default `pytest` run must not pay for them. They are
run by hand through the same pipeline, by path:

```bash
python app/mpfem/scripts/run_case.py run app/mpfem/cases-long/AdvancedPackageTransient
```

`run_case.py` resolves a bare case name under `app/mpfem/cases` alone, so a
case here is always named by its path.

## AdvancedPackageTransient

Transient electro-thermal-structural coupling in a flip-chip package: a
16 x 16 x 1.6 mm board, a copper substrate layer, a 7 x 7 x 0.34 mm underfill
holding a 3 x 3 array of solder bumps, the die's front-side copper and a
6 x 6 x 0.74 mm silicon die. Five materials, 14 domains, 79 134 elements,
1.11 M degrees of freedom, 34 adaptive steps over 100 s.
