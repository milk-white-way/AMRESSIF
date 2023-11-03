set terminal pngcairo enhanced font "Arial,12"
set output "compare.png"

# Set the delimiter to semicolon

set datafile separator ";"

# Set labels and title for the plot
set xlabel "X Axis"
set ylabel "Y Axis"

# Plot the second column from the file with a red line
plot "midline_100.txt" using 1:3 with lines lw 3 linecolor "red" title "Numerical", \
	 "midline_100.txt" using 1:4 with lines lw 2 linecolor "blue" title "Analytical"

