#!/usr/bin/gnuplot --persist

set terminal pdfcairo size 8cm, 6cm \
    transparent enhanced color \
    font "Latin Modern Math,14"

load "palettes/dark2.pal"

# set logscale x
set xrange [0:600]
# 80:150000]
# set xtics ("10^2" 100, "10^3" 1000, "10^4" 10000, "10^5" 100000)
# set xtics scale 1.5
# set ytics 0.1
# set yrange [0.1:1.05]
set title "Computing one Π(ω)"
set key top left
set xlabel "System size"
set ylabel "Time,  s"

set border lt 1 lw 1.5 lc "black" back
set grid
# set datafile separator ","

initial_model(x) = A1 + B1 * x**4
A1 = 1e-4
B1 = 0.01
final_model(x) = A2 + B2 * x**3.13
A2 = 1e-4
B2 = 0.01

fit initial_model(x) "01_goal.dat" u ($1**2):2 via A1, B1
fit final_model(x) "01_goal.dat" u ($1**2):3 via A2, B2

set output "01_goal.pdf"
plot \
    final_model(x) \
        ls 2 lw 4 notitle, \
    "01_goal.dat" \
        u ($1**2):3 with points ls 2 pt 5 ps 1 title "Final", \
    "01_goal.dat" \
        u ($1**2):3 with points pt 4 ps 1 lw 2 lc "black" notitle, \
    initial_model(x) \
        ls 3 lw 4 notitle, \
    "01_goal.dat" \
        u ($1**2):2 with points ls 3 pt 7 ps 1 title "Initial", \
    "01_goal.dat" \
        u ($1**2):2 with points pt 6 ps 1 lw 2 lc "black" notitle

set output
command = "convert -density 600 01_goal.pdf -quality 00 01_goal.png"
system command
