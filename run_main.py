import subprocess
import yaml

command_multisession = """
#!/bin/bash
source activate neurocaas
echo \"--Moving data into temporary directory and parsing path--\"
neurocaas-contrib workflow get-data-multi

datapath=$(neurocaas-contrib workflow get-datapath)
resultpath=$(neurocaas-contrib workflow get-resultpath-tmp)

echo \"--Running AutoLFADS--\"
source activate lfads-torch
python /home/ubuntu/lfads-torch/scripts/run_pbt.py $datapath {} $resultpath
source deactivate

echo "--Writing results--"
cd $resultpath/best_model
zip -r autolfads.zip *
neurocaas-contrib workflow put-result -r autolfads.zip

source deactivate
"""

command_single = """
#!/bin/bash
source activate neurocaas
echo \"--Moving data into temporary directory and parsing path--\"
neurocaas-contrib workflow get-data

datapath=$(neurocaas-contrib workflow get-datapath)
resultpath=$(neurocaas-contrib workflow get-resultpath-tmp)

echo \"--Running AutoLFADS--\"
source activate lfads-torch
python /home/ubuntu/lfads-torch/scripts/run_pbt.py $datapath {} $resultpath
source deactivate

echo "--Writing results--"
cd $resultpath/best_model
zip -r autolfads.zip *
neurocaas-contrib workflow put-result -r autolfads.zip

source deactivate
"""


def run_command_and_get_output(command):
    result = subprocess.run(['/bin/bash','-c',command], stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8').strip()


def get_config():
    command = """
    echo \"--Moving config file into temporary directory and parsing path--\"
    source activate neurocaas
    neurocaas-contrib workflow get-config
    echo $(neurocaas-contrib workflow get-configpath)
    source deactivate
    """
    return run_command_and_get_output(command).split('\n')[-1]


def main():
    configpath = get_config()
    with open(configpath, 'r') as file:
        config = yaml.safe_load(file)
        if config["multisession"] == "True":
            subprocess.run(command_multisession.format(configpath),shell=True)
        else:
            subprocess.run(command_single.format(configpath),shell=True)


if __name__ == "__main__":
    with open("/home/cbwash2/lfads-torch-fork/lfads-torch/test.txt",'w') as inf:
        inf.write("yes")
    main()