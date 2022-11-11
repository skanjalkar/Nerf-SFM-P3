import matplotlib.pyplot as plt
import pry


def plot():
    with open("./../slurm_output/slurm-37891.out", "r") as f:
        content = f.readlines()
    loss = []
    PSNR = []
    iteration = []
    for i, val in enumerate(content):
        if val[:7] == "[TRAIN]":
            x = val.split()
            for k, data in enumerate(x):
                if data == "Iter:":
                    iteration.append(int(x[k+1]))
                elif data == "Loss:":
                    loss.append(float(x[k+1][:-1]))
                elif data == "PSNR:":
                    PSNR.append(float(x[k+1]))
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(iteration, loss)
    plt.ylabel("Loss")
    plt.subplot(2, 1, 2)
    plt.xlabel("Iteration")
    plt.ylabel("PSNR")
    plt.plot(iteration, PSNR, "red", linewidth=1)
    plt.savefig("Loss-PSNR.png")
    plt.show()


if __name__ == "__main__":
    plot()