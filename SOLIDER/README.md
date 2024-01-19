# SOLIDER on [Person Attribute Recognition]

This repo provides details about how to use [SOLIDER](https://github.com/tinyvision/SOLIDER) pretrained representation on attribute recognition task.
They modify the code from [Rethinking_of_PAR](https://github.com/valencebond/Rethinking_of_PAR), and you can refer to the original repo for more details.

## Dependencies

- python 3.7
- pytorch 1.7.0
- torchvision  0.8.2
- cuda 10.1

## Datasets

- PETA: Pedestrian Attribute Recognition At Far Distance [[Paper](http://mmlab.ie.cuhk.edu.hk/projects/PETA_files/Pedestrian%20Attribute%20Recognition%20At%20Far%20Distance.pdf)][[Project](http://mmlab.ie.cuhk.edu.hk/projects/PETA.html)]

## Prepare Pre-trained Models
Step 1. Download models from [SOLIDER](https://github.com/tinyvision/SOLIDER), or use [SOLIDER](https://github.com/tinyvision/SOLIDER) to train your own models.

Step 2. Put the pretrained models under the `pretrained` file, and rename their names as `./pretrained/solider_swin_tiny(small/base).pth`

## Train Model and Get Attributes of Cloth-Changing Datasets

1. Create a directory to dowload above datasets. 
    ```
    cd SOLIDER
    mkdir data
    ```
2. Prepare datasets to have following structure:
    ```
    ${project_dir}/data
        PETA
            images/
            PETA.mat
            dataset_all.pkl
            dataset_zs_run0.pkl
    ```
3. Train baseline.
    ```
    sh run.sh
    ```  

4. Modify the dataset and model path, then run and get attributes file of different cloth-changing datasets.
    ```
    python demo_PETA_Ce.py
    python demo_PETA_last.py
    python demo_PETA_ltcc.py
    python demo_PETA_prcc.py
    ```

