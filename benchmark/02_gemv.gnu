#!/usr/bin/gnuplot --persist

set terminal pdfcairo size 8cm, 6cm \
    transparent enhanced color \
    font "Latin Modern Math,14"

load "palettes/dark2.pal"

# set logscale x
set xrange [0:4100]
set yrange [0:30]
# 80:150000]
# set xtics ("10^2" 100, "10^3" 1000, "10^4" 10000, "10^5" 100000)
# set xtics scale 1.5
# set ytics 0.1
# set yrange [0.1:1.05]
set title "Computing one Π(ω)"
set key top left
set xlabel "System size"
set ylabel "Time,  hours"

set border lt 1 lw 1.5 lc "black" back
set grid
# set datafile separator ","
set fit quiet

initial_model(x) = A1 + B1 * x**4
A1 = 1e-4
B1 = 0.01
fit initial_model(x) "01_goal.dat" u ($1**2):($2 / 3600.0) via A1, B1

B2 = (26.0 - A1) / 4096.0**4
print B1, B2

set output "02_gemv.pdf"
plot \
    initial_model(x) \
        ls 3 lw 4 notitle, \
    "01_goal.dat" \
        u ($1**2):($2 / 3600.0) with points ls 3 pt 7 ps 1 title "Initial", \
    "01_goal.dat" \
        u ($1**2):($2 / 3600.0) with points pt 6 ps 1 lw 2 lc "black" notitle, \
    A1 + B2 * x**4 \
        ls 4 lw 4 notitle, \
    "plasmons_in_fractals.dat" \
        u 1:2 with points ls 4 pt 5 ps 1 title "Current", \
    "plasmons_in_fractals.dat" \
        u 1:2 with points pt 4 ps 1 lw 2 lc "black" notitle

set output
command = "convert -density 600 02_gemv.pdf -quality 00 02_gemv.png"
system command

