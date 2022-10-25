import os
import pry

def createMatchestxt(path):

    for i in range(1, 5):
        file_path = path+f"matching{i}.txt"
        with open(file_path, 'r')  as f:
            lines = f.readlines()

        for line in lines[1:]:
            nums = line.split()
            nFeatures = nums[0]
            otherImages = nums[6:] # 1-3 is rgb value, 4,5 is the current image point value, 6 onwards is the corresponding image point

            for idx, imgPt in enumerate(otherImages):
                if idx%3 == 0:
                    with open(path+f'matches{i}{imgPt}.txt', 'a') as m:
                        savePt = f'{nums[4]}' + " " + f'{nums[5]}' + " " + f"{otherImages[idx+1]}" + " " + f'{otherImages[idx+2]}' + " " + f'{nums[1]}' + " " + f'{nums[2]}' + " " + f'{nums[3]}' + "\n"
                        m.write(savePt)


