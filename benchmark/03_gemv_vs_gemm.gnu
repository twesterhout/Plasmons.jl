#!/usr/bin/gnuplot --persist

set terminal pdfcairo size 8cm, 6cm \
    transparent enhanced color \
    font "Latin Modern Math,14"

load "palettes/dark2.pal"

# set logscale x
# set xrange [0:00]
set yrange [0:150]
# 80:150000]
# set xtics ("10^2" 100, "10^3" 1000, "10^4" 10000, "10^5" 100000)
# set xtics scale 1.5
# set ytics 0.1
# set yrange [0.1:1.05]
# set key top left
# set logscale x
set xlabel "N = M = K"
set ylabel "GFLOPS"

set border lt 1 lw 1.5 lc "black" back
# set grid lw 2
# set datafile separator ","
# set fit quiet

# initial_model(x) = A1 + B1 * x**4
# A1 = 1e-4
# B1 = 0.01
# fit initial_model(x) "01_goal.dat" u ($1**2):($2 / 3600.0) via A1, B1

# B2 = (26.0 - A1) / 4096.0**4
# print B1, B2

flops(n, t) = n * n * n * 2.0 / t / 1e9

basename = "03_gemv_vs_gemm"
set output basename . ".pdf"
plot \
    "03_gemv_vs_gemm.dat" \
        u 1:(flops($1, $2)) with lines ls 3 lw 4 notitle, \
    "" \
        u 1:(flops($1, $2)) with points ls 3 pt 7 ps 1 title "Matrix-matrix", \
    "" \
        u 1:(flops($1, $2)) with points pt 6 ps 1 lw 2 lc "black" notitle, \
    "" \
        u 1:(flops($1, $3)) with lines ls 4 lw 4 notitle, \
    "" \
        u 1:(flops($1, $3)) with points ls 4 pt 5 ps 1 title "Matrix-vector", \
    "" \
        u 1:(flops($1, $3)) with points pt 4 ps 1 lw 2 lc "black" notitle

set output
command = "convert -density 600 " . basename . ".pdf -quality 00 " . basename . ".png"
system command

