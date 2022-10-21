import argparse
import numpy as np

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--cal_path', default='Data/calibration.txt', help='Path to intrinsic matrix')
    Args = Parser.parse_args()
    cal_path = Args.cal_path

    with open(cal_path, 'r') as f:
        contents = f.read()

    K = []
    for i in contents.split():
        K.append(float(i))

    K = np.array(K)
    K = np.reshape(K, (3,3))



if __name__ == "__main__":
    main()