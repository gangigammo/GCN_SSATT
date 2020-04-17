a=0
while [ $a -lt 11 ]
do
    #a=`expr $a + 1`
    b=`echo "scale=5; $a / 10.0" | bc`
    echo $b


    #############add epoch and validation
    python3 train_subatt.py --random --n_train 6 --patience 100 --epoch 100000 --alpha $b --times 300 --dataset citeseer
    a=`expr $a + 1`
done
echo "" >> log.txt


a=0
while [ $a -lt 11 ]
do
    #a=`expr $a + 1`
    b=`echo "scale=5; $a / 10.0" | bc`
    echo $b


    #############add epoch and validation
    python3 tmp_exp.py --random --n_train 6 --patience 100 --epoch 100000 --alpha $b --times 300 --dataset citeseer
    a=`expr $a + 1`
done
echo "" >> log.txt





a=0
while [ $a -lt 11 ]
do
    #a=`expr $a + 1`
    b=`echo "scale=5; $a / 10.0" | bc`
    echo $b


    #############add epoch and validation
    #python3 train_subatt.py --random --n_train 4 --patience 100 --epoch 100000 --alpha $b --times 300 --dataset cora
    #python3 tmp_exp.py --random --n_train 4 --patience 100 --epoch 100000 --alpha $b --times 300 --dataset cora
    a=`expr $a + 1`
done
echo "" >> log.txt



#a=0
#while [ $a -lt 11 ]
#do
#    #a=`expr $a + 1`
#    b=`echo "scale=5; $a / 10.0" | bc`
#    echo $b
#    #python3 train_subatt.py --random --n_train 4 --patience 10 --epoch 200 --alpha $b --times 300 --dataset cora
#    #python3 tmp_exp.py --random --n_train 4 --patience 10 --epoch 200 --alpha $b --times 300 --dataset cora
#    a=`expr $a + 1`
#done
#echo "" >> log.txt





a=1
while [ $a -lt 11 ]
do
    #a=`expr $a + 1`
    b=`echo "scale=5; $a / 10.0" | bc`
    echo $b
    #python3 train_subatt.py --random --n_train 4 --patience 10 --epoch 200 --alpha $b --times 300 --dataset cora
    #python3 train_subatt.py --random --n_train 6 --patience 10 --epoch 200 --alpha $b --times 300 --dataset citeseer
    #python3 train_subatt.py --random --n_train 7 --patience 10 --epoch 200 --alpha $b --times 300 --dataset pubmed
    a=`expr $a + 1`
done
echo "" >> log.txt

a=0
while [ $a -lt 11 ]
do
    #a=`expr $a + 1`
    b=`echo "scale=5; $a / 10.0" | bc`
    echo $b
    #python3 train_subatt.py --random --n_train 4 --patience 10 --epoch 200 --alpha $b --times 300 --dataset cora
    #python3 tmp_exp.py --random --n_train 6 --patience 10 --epoch 200 --alpha $b --times 300 --dataset citeseer
    #python3 tmp_exp.py --random --n_train 7 --patience 10 --epoch 200 --alpha $b --times 300 --dataset pubmed
    a=`expr $a + 1`
done
echo "" >> log.txt




#a=0
#while [ $a -lt 5 ]
#do
#    a=`expr $a + 1`
#    b=`expr 6 \* $a`
#    python3 train_subatt.py --random --n_train $b --patience 10 --epoch 200 --alpha 0.5 --times 300 --dataset citeseer
#    python3 tmp_exp.py --random --n_train $b --patience 10 --epoch 200 --alpha 0 --times 300 --dataset citeseer
#    python3 tmp_exp.py --random --n_train $b --patience 10 --epoch 200 --alpha 0.5 --times 300 --dataset citeseer
#    echo "" >> log.txt
#done
#echo "" >> log.txt


#a=0
#while [ $a -lt 5 ]
#do
#    a=`expr $a + 1`
#    b=`expr 7 \* $a`
#    python3 train_subatt.py --random --n_train $b --patience 10 --epoch 200 --alpha 0.5 --times 300 --dataset pubmed
#    python3 tmp_exp.py --random --n_train $b --patience 10 --epoch 200 --alpha 0 --times 300 --dataset pubmed
#    python3 tmp_exp.py --random --n_train $b --patience 10 --epoch 200 --alpha 0.5 --times 300 --dataset pubmed
#    echo "" >> log.txt
#done
#echo "" >> log.txt
