# Nerf-SFM-P3
Implementation of Structure from Motion and NeRF paper in pytorch.

## Authors
[Shreyas Kanjalkar](https://github.com/zen1405)\
Khizar Mohammed Amjed Mohammed

# Phase 1

In order to run the code, navigate to Phase1
```cd Phase1/```
Then run the following command,

```python3 Wrapper.py --findImgPair=True```



# Phase 2

Phase 2 is the implementation of the original nerf paper using pytorch. In order to run the code, please go to
```cd Phase2/```

and then run

```python3 main.py``` followed by ```python3 plot.py``` to run the network and plot the graph of loss, PSNR vs iterations

Details about the implementation and paper can be found in the report. Results from NeRF on the lego set are shown here. The training was done on
WPI's turing cluster. Details about that can be found in the job_script.sh

Here are some images which show the model rendered for particular number of iterations.

* 1000 Iterations:\
![1000.png](https://github.com/zen1405/Nerf-SFM-P3/blob/main/Images/1000.png)

* 10000 Iterations:\
![10000.png](https://github.com/zen1405/Nerf-SFM-P3/blob/main/Images/10000.png)

* 25000 Iterations:\
![25000.png](https://github.com/zen1405/Nerf-SFM-P3/blob/main/Images/25000.png)

* 50000 Iterations:\
![50000.png](https://github.com/zen1405/Nerf-SFM-P3/blob/main/Images/50000.png)

* 100000 Iterations:\
![100000.png](https://github.com/zen1405/Nerf-SFM-P3/blob/main/Images/100000.png)

* 150000 Iterations:\
[!150000.png](https://github.com/zen1405/Nerf-SFM-P3/blob/main/Images/150000.png)

* 200000 Iterations:\
[!200000.png](https://github.com/zen1405/Nerf-SFM-P3/blob/main/Images/200000.png)

The video for the lego model on number of iterations can be found [here](https://drive.google.com/drive/folders/1-a3yk7HnIo5qxqlLkmGf6Q0PxMG4eBAA)