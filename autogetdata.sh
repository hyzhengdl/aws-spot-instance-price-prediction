#!/bin/bash
cd /Users/wxc575843/Documents/papers/fxu/
datapath=/Users/wxc575843/Documents/papers/fxu/autodata/
if [ ! -d "./autodata" ]; then
  mkdir ./autodata
fi

vare=$(date "+%Y-%m-%dT00:00:00")
varb=$(date -v -90d +%Y-%m-%dT00:00:00)

regions=(us-east-1 ap-southeast-1)
instances=(m4.xlarge d2.8xlarge g2.2xlarge m3.medium r3.large)
for region in ${regions[@]}
do
	cd $datapath
	if [ ! -d $region ]; then
  		mkdir $region
	fi
	for instance in ${instances[@]}
	do
		cd $datapath
		cd $region
		if [ ! -d $instance ]; then
  			mkdir $instance
		fi
		cd $instance
		if [ ! -f $instance ]; then
			touch $instance
		fi
		ec2-describe-spot-price-history -H --region $region --instance-type $instance --start-time $varb --end-time $vare --product-description "Linux/UNIX" > tmp
		tail +2 tmp > tmp2
		cat tmp2 $instance > tmp
		rm tmp2 $instance
		mv tmp $instance
	done
done