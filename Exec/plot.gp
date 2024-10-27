set terminal pngcairo enhanced size 1000,800 font "Arial,16"
set output "lid_driven_vertical_compare_re1000.png"

# Set the delimiter to semicolon

set datafile separator ","

# Set labels and title for the plot
set xlabel "y"
set ylabel "ucont_x"

# Plot the second column from the file with a red line
#plot "midline_100.txt" using 1:3 with lines lw 3 linecolor "red" title "Numerical", \
#	 "midline_100.txt" using 1:4 with lines lw 2 linecolor "blue" title "Analytical"
plot "re1000_vertical.txt" using 1:2 with points pt 8 title "Ghia et al"

