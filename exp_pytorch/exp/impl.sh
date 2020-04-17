a=0
while [ $a -lt 5 ]
do
    a=`expr $a + 1`
    b=`expr 4 \* $a`
    python3 train_subatt.py --random --n_train $b --patience 10 --epoch 200 --alpha 0.6 --times 300 --dataset cora
done
echo "" >> log.txt

a=0
while [ $a -lt 5 ]
do
    a=`expr $a + 1`
    b=`expr 4 \* $a`
    python3 tmp_exp.py --random --n_train $b --patience 10 --epoch 200 --alpha 0.4 --times 300 --dataset cora
done
echo "" >> log.txt

a=0
while [ $a -lt 5 ]
do
    a=`expr $a + 1`
    b=`expr 6 \* $a`
    python3 train_subatt.py --random --n_train $b --patience 10 --epoch 200 --alpha 0.6 --times 300 --dataset citeseer
done
echo "" >> log.txt

a=0
while [ $a -lt 5 ]
do
    a=`expr $a + 1`
    b=`expr 6 \* $a`
    python3 tmp_exp.py --random --n_train $b --patience 10 --epoch 200 --alpha 0.5 --times 300 --dataset citeseer
done
echo "" >> log.txt


#a=0
#while [ $a -lt 5 ]
#do
#    a=`expr $a + 1`
#    b=`expr 7 \* $a`
#    python3 train_subatt.py --random --n_train $b --patience 10 --epoch 200 --alpha 0.3 --times 300 --dataset pubmed
#done
#echo "" >> log.txt
#
#a=0
#while [ $a -lt 5 ]
#do
#    a=`expr $a + 1`
#    b=`expr 7 \* $a`
#    python3 tmp_exp.py --random --n_train $b --patience 10 --epoch 200 --alpha 0.8 --times 300 --dataset pubmed
#done
#echo "" >> log.txt
