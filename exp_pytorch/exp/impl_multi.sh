k=0
while [ $k -lt 10 ]
do
  bash impl.sh
  echo "break" >> log.txt
done