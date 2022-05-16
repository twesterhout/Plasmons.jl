#!/usr/bin/gnuplot --persist

set terminal pdfcairo size 8cm, 6cm \
    transparent enhanced color \
    font "Latin Modern Math,14"

load "palettes/dark2.pal"

# set logscale x
set xrange [0:7500]
set yrange [0:1]
# 80:150000]
# set xtics ("10^2" 100, "10^3" 1000, "10^4" 10000, "10^5" 100000)
# set xtics scale 1.5
# set ytics 0.1
# set yrange [0.1:1.05]
set title "Computing one Π(ω)"
set key top right
set xlabel "System size"
set ylabel "Time,  hours"

set border lt 1 lw 1.5 lc "black" back
set grid
# set datafile separator ","
set fit quiet

_model(x, A, B) = A + B * x**4
A1 = 1e-4
B1 = 0.01
fit _model(x, A1, B1) "01_goal.dat" u ($1**2):($2 / 3600.0) via A1, B1

A2 = A1
B2 = (26.0 - A1) / 4096.0**4

A3 = A2
B3 = B2
fit _model(x, A3, B3) "04_batched.dat" u ($1**2):($2 / 3600.0) via A3, B3

A4 = A3
B4 = B3
fit _model(x, A4, B4) "05_gpu.dat" u ($1**2):($2 / 3600.0) via A4, B4

print B1, B2, B3, B4

set output "05_gpu.pdf"
plot \
    _model(x, A1, B1) \
        ls 3 lw 4 title "Naive", \
    _model(x, A2, B2) \
        ls 3 lw 4 title "Matrix-vector", \
    _model(x, A3, B3) \
        ls 4 lw 4 title "Matrix-matrix", \
    _model(x, A4, B4) \
        ls 2 lw 4 title "Current", \
    "05_gpu.dat" \
        u ($1**2):($2 / 3600.0) with points ls 2 pt 13 ps 1.2 notitle, \
    "05_gpu.dat" \
        u ($1**2):($2 / 3600.0) with points pt 12 ps 1.2 lw 2 lc "black" notitle

set output
command = "convert -density 600 05_gpu.pdf -quality 00 05_gpu.png"
system command

