a=0
while [ $a -lt 5 ]
do
    a=`expr $a + 1`
    b=`expr 4 \* $a`
    python3 tmp_exp.py --random --n_train $b --patience 10 --epoch 200 --alpha 0.4 --times 300 --dataset cora --struc_feat betweeness
    python3 tmp_exp.py --random --n_train $b --patience 10 --epoch 200 --alpha 0.4 --times 300 --dataset cora --struc_feat closeness
    python3 tmp_exp.py --random --n_train $b --patience 10 --epoch 200 --alpha 0.4 --times 300 --dataset cora --struc_feat degree
    python3 tmp_exp.py --random --n_train $b --patience 10 --epoch 200 --alpha 0.4 --times 300 --dataset cora --struc_feat eigen
    python3 tmp_exp.py --random --n_train $b --patience 10 --epoch 200 --alpha 0.4 --times 300 --dataset cora --struc_feat pagerank
done
echo "" >> log.txt
