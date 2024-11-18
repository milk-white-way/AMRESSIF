set terminal pngcairo enhanced size 1000,800 font "Arial,24"
set output "test.png"

# Set the delimiter to semicolon

set datafile separator ","

# Set labels and title for the plot
set xlabel "x"
set ylabel "ucont-y"

# Plot the second column from the file with a red line
plot "./book2.csv" using 1:8 with points pt 7 ps 3 title "Ghia et al", \
    "./ren10000_horizontal_numel40000.csv" using 1:3 with lines lw 3 title "AMRESSIF 128x128"

#plot "./book1.csv" using 1:3 with points pt 7 ps 3 title "Ghia et al", \
#    "./ren400_vertical_numel40000.csv" using 2:3 with lines lw 3 title "AMRESSIF 128x128"

#"./matlabRK_128_2e-3_1000_re100_ucontx_on_vertical_line.txt" using 1:2 with lines lw 3 title "MATLAB RK 128x128", \

