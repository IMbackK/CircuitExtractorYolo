
set title "Occuances of Models"
set yrange [0:40]
set boxwidth 0.5
set datafile separator ','
set xtics rotate by 90 right
set style fill solid
set nokey
plot "statistics.txt" using 1:3:xtic(2) lc "blue" with boxes
