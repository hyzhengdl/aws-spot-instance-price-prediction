#! /bin/bash
declare -a DIRS
declare -a FILES
DIRS=(d28xlarge g22xlarge m4xlarge r3large m3medium)
# dirs=(m3medium)
FILES=(a c d e)
for dir in ${DIRS[*]}
do
    python gen.py $dir || exit
    for file in ${FILES[*]}
    do
        echo $dir
        echo $file
        if [ $dir = m3medium -a $file = e ]; then
            continue
        fi
        python train2.py $dir $file || exit
    done
done
