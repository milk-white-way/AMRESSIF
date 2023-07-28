rm ./backtrace_line_address.log

cat Backtrace.0 | grep "./main2d.gnu.DEBUG.MPI.ex" | awk '{print $3}' | awk 'sub(/^.{1}/,"")' | sed 's/.$//' >> backtrace_line_address.log

#cat ./backtrace_line_address.log

while read p; do 
  echo "========== Printing error for address line [$p] =========="
  addr2line -Cpfie main2d.gnu.DEBUG.MPI.ex "$p"
done < ./backtrace_line_address.log
