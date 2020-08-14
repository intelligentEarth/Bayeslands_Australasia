
#!/bin/bash  
echo Running Revamped 	 

problem=1
replica=4

samples=56
swapint=2
maxtemp=2  
burn=0.1
pt_stage=0.1
raintimeint=4
initialtopoep=0.5 # not used anymore
cov=0
inittopo=0
uplift=0


  
python PT.py -p $problem -s $samples -r $replica -t $maxtemp -swap $swapint -b $burn -pt $pt_stage  -epsilon $initialtopoep -rain_intervals $raintimeint -inittopo $inittopo -uplift $uplift
# python visualise.py -p $problem -s $samples -r $replica -t $maxtemp -swap $swapint -b $burn -pt $pt_stage  -epsilon $initialtopoep -rain_intervals $raintimeint -inittopo $inittopo -uplift $uplift
  

