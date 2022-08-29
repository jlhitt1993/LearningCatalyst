# Python script to analyze the images created during high-throughput catalyst screening
# Author: Jeremy Hitt
# When selecting points, start from the top and move down and right. Once you reach the end of a row,
# start at the top left of the next row over and continue in the same direction as before.
import math
#import matplotlib.pyplot as plt
#from time import sleep
import pickle
import cv2
import numpy as np
import os
import plotly.express as px
import plotly.io as pio
import pandas as pd
from tkinter import Tk, filedialog
#from PIL import ImageTk, Image
from numba import jit, cuda
import wx
import wx.lib.scrolledpanel
from tqdm import tqdm
All_Circles = np.zeros((933, 1400), np.uint8)
image = []
im_diff = []
Total = []
Background = []
Points = []
Rings = []
Mini = []
diff_range = 1
clicks = 0
nums = []
#gas = ""


class SimpleFrame(wx.Frame):
    def __init__(self, parent):
        super(SimpleFrame, self).__init__(parent)
        # add a panel so it looks the correct on all platforms
        self.frame_panel = wx.Panel(self)
        frame_panel = self.frame_panel
        # image panel
        self.image_panel = wx.lib.scrolledpanel.ScrolledPanel(frame_panel, style=wx.SIMPLE_BORDER)
        image_panel = self.image_panel
        image_panel.SetAutoLayout(True)
        image_panel.SetupScrolling()
        # image panel - image control
        self.image_ctrl = wx.StaticBitmap(image_panel)
        self.image_ctrl.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        win = Tk()
        self.img = wx.Image(filedialog.askopenfilename(), wx.BITMAP_TYPE_ANY)
        self.image_ctrl.SetBitmap(wx.Bitmap(self.img))
        image_panel.Layout()
        image_sizer = wx.BoxSizer(wx.VERTICAL)
        image_sizer.Add(self.image_ctrl)
        image_panel.SetSizer(image_sizer)
        # frame sizer
        frame_sizer = wx.BoxSizer(wx.HORIZONTAL)
        frame_sizer.Add(image_panel, proportion=1, flag=wx.EXPAND | wx.ALL)
        frame_panel.SetSizer(frame_sizer)
        win.destroy()
        return

    def OnLeftDown(self, event):
        global Total, clicks, Points, Rings
        ctrl_pos = event.GetPosition()
        points, rings = get_points(ctrl_pos.x, ctrl_pos.y, 40)
        print(ctrl_pos.x, ctrl_pos.y)
        clicks += 1
        print(clicks)
        Points.append(np.where(points == 1))
        Rings.append(np.where(rings == 1))
        #print("ctrl_pos: " + str(ctrl_pos.x) + ", " + str(ctrl_pos.y))


"""
def draw_circle(event, x, y, f, p):
    global All_Circles, Total
    if event == cv2.EVENT_LBUTTONDOWN:
        circle = np.zeros((933, 1400), np.uint8)
        cv2.circle(circle, (x, y), 15, 255, -1)
        cv2.circle(All_Circles, (x, y), 15, 255, -1)
        points = np.transpose(np.where(circle == 255))
        cv2.imshow("output", cv2.add(All_Circles, image[0]))
        Points.append(points)
"""


@jit
def get_points(center_x, center_y, radius):
    points2 = np.zeros((4000, 6000), np.uint8)
    ring = np.zeros((4000, 6000), np.uint8)
    # get the pixels that are located inside the circle and ring
    for col in range(4000):
        for row in range(6000):
            # calc the distance each pixel is from the center of the circle
            square_dist = (center_x - row) ** 2 + (center_y - col) ** 2
            if square_dist <= radius ** 2:
                points2[col, row] = 1
            if (square_dist <= (radius+90) ** 2) and (square_dist >= (radius+45) ** 2):
                ring[col, row] = 1
    return points2, ring


def read_image(file, bckgnd):
    image2 = cv2.imread(file)
    (B, G, R) = cv2.split(image2)
    g = cv2.addWeighted(R, 2.7, R, 0, 0)
    return g


def get_diff(im1, im2):
    diff = cv2.subtract(im2, im1)*5
    diff = diff.astype('uint8')
    return diff


@jit
def get_ave_points(pic_points, ryngs, img):
    # loop through 66 points
    total, bckgnd2 = [], []
    for j in range(len(pic_points)):
        su = 0.0
        bckgnd = 0.0
        # loop through each pixel in each point
        for i in range(len(pic_points[j][0])):
            su += int(img[pic_points[j][0][i], pic_points[j][1][i]])
        su = su / len(pic_points[j][0])
        for k in range(len(ryngs[j][0])):
            bckgnd += int(img[ryngs[j][0][k], ryngs[j][1][k]])
        bckgnd = bckgnd / len(ryngs[j][0])
        total.append(su)
        bckgnd2.append(bckgnd)
    return total, bckgnd2


def light_up(I, Q):
    global Total, Background, Folder, count
    lightup = np.array([np.inf for _ in range(len(im_diff))])
    for J in range(len(Total)):  # range 0-15
        ring_diff = Background[J][I] - Background[0][I]
        #print(Total[J][I])
        if (Total[J][I] - ring_diff) > Q * Total[0][I]:
            lightup[J] = J
            break
    #print(lightup)
    return np.min(lightup)


def generate_percents():
    PCT = np.zeros((3, 66))
    counter = 0
    for z in range(11):
        for j in range(11 - z):
            PCT[:, counter] = [z / 10.0, (10 - z - j) / 10.0, j / 10.0]
            counter += 1
    return pd.DataFrame(PCT.T, columns=["Percent A", "Percent B", "Percent C"])


def plot_array(pct, mini, folder, miN, maX, null):
    #global gas
    array = pd.concat((pct[:len(mini)], pd.DataFrame(mini, index=pct.index[:len(mini)],
                                                     columns=["Onset potential vs Ag/AgCl"])), axis=1)
    size = 20*(array["Onset potential vs Ag/AgCl"]+1.8)
    for a in range(len(size)):
        if math.isnan(size[a]):
            size[a] = 4
    """
    if gas == 'Ar':
        maX = 1.3
        miN = 0.5
        null = 1.7
    elif gas == 'H2':
        maX = 0.6
        miN = -0.2
        null = 1.0
    else:
        maX = 1.3
        miN = -0.2
        null = 1.7
    """
    fig = px.scatter_ternary(array, a="Percent B", b="Percent A", c="Percent C", range_color=[miN, maX],
                             color="Onset potential vs Ag/AgCl", size=size,
                             size_max=18, color_continuous_scale="rainbow", width=900, height=700)
    fig.show()
    save_name = input("Would you like to save the results? y or n: ")
    if save_name == "Y" or save_name == "y":
        pio.write_image(fig, os.path.join(folder, "Results.png"), height=500, width=850, scale=2.5)
        Onset = pd.Series(mini)
        Onset.replace(np.nan, null, inplace=True)
        Onset.to_excel(os.path.join(folder, "Onset V.xlsx"))


def voltages(s):
    return float(s[3:-6])


if __name__ == '__main__':
    root = Tk()
    Folder = filedialog.askdirectory()
    gas = Folder[-2:]
    root.destroy()
    fyles = []
    for j in os.listdir(Folder):
        if j[-5:] == "V.tif":
            fyles.append(j)
    Files = sorted(fyles, key=voltages)
    pyckle = False
    if 'circles.pkl' in os.listdir(Folder):
        Points, Rings = pickle.load(open(os.path.join(Folder, 'circles.pkl'), 'rb'))
        pyckle = True
    else:
        app = wx.App()
        frame = SimpleFrame(None)
        frame.Show()
        app.MainLoop()
    for count in range(len(Files)):
        if Files[count][-5:] == "V.tif":
            image.append(read_image(os.path.join(Folder, Files[count]), False))
            print(os.path.join(Folder, Files[count]))
            print(count + 1, " of ", len(Files))
    if not pyckle:
        f = open(os.path.join(Folder, 'circles.pkl'), "wb")
        pickle.dump([Points, Rings], f)
        f.close()
    for g in range(len(image)-diff_range):
        im_diff.append(get_diff(image[0], image[g+diff_range]))
        nums.append(float(Files[g + diff_range].split()[1]))
        if not pyckle:
            print("saving ", g+diff_range, " out of ", len(image)-diff_range)
            cv2.imwrite(os.path.join(Folder, str(g+diff_range) + Files[g+diff_range][:-4]+'_diff.tiff'), im_diff[g])
    Q = input("Enter the multiplication factor: ")
    #nums = input("Enter the first and last voltage of the scan and the null value separated by a space: ")
    #miN, maX, null = nums.split()
    miN = np.min(nums)
    maX = np.max(nums)
    null = maX + 0.4
    #print(miN, maX, null)
    for r in tqdm(range(len(im_diff))):
        T, B = get_ave_points(tuple(Points), tuple(Rings), im_diff[r])
        # Total has length equal to the number of files in a folder
        Total.append(T)
        Background.append(B)

    # Uncomment the section below to visualize the mask
    """
    temp = np.zeros((4000, 6000), np.uint8)
    for t in range(len(Rings)):
        temp[Rings[t]] = 1
        temp[Points[t]] = 1
    plt.imshow(temp)
    plt.show()
    """

    for i in range(len(Total[0])):
        Mini.append(light_up(i, float(Q)))   # Mini has length equal to the number of points and the values are the index of the
                                   # index corresponding to when the spot first lit up.
    files = [sorted(os.listdir(Folder))[-1]] + sorted(os.listdir(Folder))[:-1]
    for i in range(len(Mini)):
        if Mini[i] == np.inf:
            Mini[i] = np.nan
        else:
            Mini[i] = float(Files[int(Mini[i])].split()[1])
    #print(Mini)
    Pct = generate_percents()
    plot_array(Pct, Mini, Folder, miN, maX, null)

