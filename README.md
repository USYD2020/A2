# COMP5349 A2

Setting your EMR with one Master node and four core node and choose all the instance type as c4.xlarge

After login your EMR system First install git on your EMR system: sudo yum install git

The using git clone to clone the dataset, .py file and .sh file from my github address: git clone
[https://github.com//COMP5349.git](https://github.com/USYD2020/A2.git)

Put the dataset json files on to hdfs, then change the test_data variable according to the json file path on line 145. hdfs dfs -put "your data file path on your current path"

After the configuration, just start the program by type "./a2.sh" on the command line to run the script.

A result output file will be create on your current path.
