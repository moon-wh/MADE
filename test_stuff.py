from data.datasets.ccvid import CCVID

def main():
    cc = CCVID(
            root="/home/c3-0/datasets/CCVID/", 
            sampling_step=64, 
            seq_len=8, 
            stride=4, 
            num_seq=4, 
            transform=None, 
            is_train=True, 
            meta_dir='/home/sriniana/projects/MADE/SOLIDER/PAR_PETA_105_ccvid.txt', 
            meta_dims=105
        )
    print(len(cc.train[1][0]))


if __name__ == "__main__":
    main()