#!/usr/bin/gnuplot --persist

set terminal pdfcairo size 8cm, 6cm \
    transparent enhanced color \
    font "Latin Modern Math,14"

load "palettes/dark2.pal"

# set logscale x
set xrange [0:10000]
set yrange [0:1]
# 80:150000]
set xtics (0, 2000, 4000, 6000, 8000, 10000)
# set xtics scale 1.5
# set ytics 0.1
# set yrange [0.1:1.05]
# set key top right
unset key
set title "Computing one Π(ω)"
set xlabel "System size"
set ylabel "Time,  hours"

set border lt 1 lw 1.5 lc "black" back
set grid
# set datafile separator ","
set fit quiet

_model(x, A, B) = A + B * x**4
A1 = 1e-9
B1 = 0.001
fit _model(x, A1, B1) "01_goal.dat" u ($1**2):($2 / 3600.0) via A1, B1

A2 = A1
B2 = (26.0 - A1) / 4096.0**4

A3 = A2
B3 = B2
fit _model(x, A3, B3) "04_batched.dat" u ($1**2):($2 / 3600.0) via A3, B3

A4 = 1e-15
B4 = 0.001
fit _model(x, A4, B4) "05_gpu.dat" u ($1**2):($2 / 3600.0) via A4, B4

A5 = A4
B5 = B4
fit _model(x, A5, B5) "06_sparse.dat" u ($1**2):($2 / 3600.0) via A5, B5

A6 = A5
B6 = B5
fit _model(x, A6, B6) "07_final.dat" u ($1**2):($2 / 3600.0) via A6, B6

_model2(x, A, B, nu) = A + B * x**nu
A7 = A6
B7 = B6
nu = 4
fit _model2(x, A7, B7, nu) "07_final.dat" u ($1**2):($2 / 3600.0) via A7, B7, nu
print A7, B7, nu

print B1, B2, B3, B4, B5, B6

print _model(10000, A1, B1), _model(10000, A6, B6)
print _model(10000, A1, B1)/_model(10000, A6, B6)

set output "07_final.pdf"
plot \
    _model(x, A1, B1) \
        ls 3 lw 4 title "Naive", \
    _model(x, A2, B2) \
        ls 3 lw 4 title "Matrix-vector", \
    _model(x, A3, B3) \
        ls 3 lw 4 title "Matrix-matrix", \
    _model(x, A4, B4) \
        ls 3 lw 4 title "Nvidia V100", \
    _model(x, A5, B5) \
        ls 3 lw 4 title "Nvidia V100 + sparse", \
    _model(x, A6, B6) \
        ls 2 lw 4 title "Nvidia A100 + everything", \
    "07_final.dat" \
        u ($1**2):($2 / 3600.0) with points ls 2 pt 13 ps 1.2 notitle, \
    "07_final.dat" \
        u ($1**2):($2 / 3600.0) with points pt 12 ps 1.2 lw 2 lc "black" notitle

set output
command = "convert -density 600 07_final.pdf -quality 00 07_final.png"
system command

