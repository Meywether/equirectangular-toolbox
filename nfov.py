# Copyright 2017 Nitish Mutha (nitishmutha.com)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import pi
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class NFOV():
    def __init__(self, height=400, width=800):
        self.FOV = [0.5, 0.5]
        self.PI = pi
        self.PI_2 = pi * 0.5
        self.PI2 = pi * 2.0
        self.height = height
        self.width = width
        self.screen_points = self._get_screen_img()

    def _get_coord_rad(self, isCenterPt, center_point=None):
        return (center_point * 2 - 1) * np.array([self.PI, self.PI_2]) \
            if isCenterPt \
            else \
            (self.screen_points * 2 - 1) * np.array([self.PI, self.PI_2]) * (
                np.ones(self.screen_points.shape) * self.FOV)

    def _get_coord_rad_customData(self, customCoords):
        return  (customCoords * 2 - 1) * np.array([self.PI, self.PI_2]) * (np.ones(customCoords.shape) * self.FOV)

    def _get_screen_img(self): # screen = ergebnisbild
        xx, yy = np.meshgrid(np.linspace(0, 1, self.width), np.linspace(0, 1, self.height))
        return np.array([xx.ravel(), yy.ravel()]).T

    def _calcSphericaltoGnomonic(self, convertedScreenCoord):
        x = convertedScreenCoord.T[0]
        y = convertedScreenCoord.T[1]

        rou = np.sqrt(x ** 2 + y ** 2)
        c = np.arctan(rou)
        sin_c = np.sin(c)
        cos_c = np.cos(c)

        lat = np.arcsin(cos_c * np.sin(self.cp[1]) + (y * sin_c * np.cos(self.cp[1])) / rou)
        lon = self.cp[0] + np.arctan2(x * sin_c, rou * np.cos(self.cp[1]) * cos_c - y * np.sin(self.cp[1]) * sin_c)


        lat = (lat / self.PI_2 + 1.) * 0.5 #lat = zw. 0 und 1 für linear mapping auf pixel im nächsten Schritt
        lon = (lon / self.PI + 1.) * 0.5

        return np.array([lon, lat]).T

    def _bilinear_interpolation(self, screen_coord):

        # Hier wird das kleine Bild auf einen großen Raster projeziert. Bzw. hier wird der raster vorbereitet

        # Liefert Rest zurück und mappt damit linear auf die pixel
        uf = np.mod(screen_coord.T[0],1) * self.frame_width  # long - width
        vf = np.mod(screen_coord.T[1],1) * self.frame_height  # lat - height

        x0 = np.floor(uf).astype(int)
        y0 = np.floor(vf).astype(int)
        x2 = np.add(x0, np.ones(uf.shape).astype(int))
        y2 = np.add(y0, np.ones(vf.shape).astype(int))

        base_y0 = np.multiply(y0, self.frame_width)
        base_y2 = np.multiply(y2, self.frame_width)

        # https://en.wikipedia.org/wiki/Bilinear_interpolation
        # A - D -> sind bekannte Punkte dadurch vereinfacht sich die Formel. Die Punkte A - D sind alle Eckpunkte des Bildes. Eventuell sogar zwischen 0 und 1 skaliert.

        A_idx = np.add(base_y0, x0)
        B_idx = np.add(base_y2, x0)
        C_idx = np.add(base_y0, x2)
        D_idx = np.add(base_y2, x2)

        flat_img = np.reshape(self.frame, [-1, self.frame_channel]) # Hier stecken die farbinformationen drinnen
        # import matplotlib.pyplot as plt
        # plt.imshow(flat_img)
        # plt.show()


        A = np.take(flat_img, A_idx, axis=0)
        B = np.take(flat_img, B_idx, axis=0)
        C = np.take(flat_img, C_idx, axis=0)
        D = np.take(flat_img, D_idx, axis=0)

        wa = np.multiply(x2 - uf, y2 - vf)
        wb = np.multiply(x2 - uf, vf - y0)
        wc = np.multiply(uf - x0, y2 - vf)
        wd = np.multiply(uf - x0, vf - y0)

        # interpolate
        AA = np.multiply(A, np.array([wa, wa, wa]).T)
        BB = np.multiply(B, np.array([wb, wb, wb]).T)
        CC = np.multiply(C, np.array([wc, wc, wc]).T)
        DD = np.multiply(D, np.array([wd, wd, wd]).T)
        # Hier werden die Farben und Werte auf das neue Bild gemappt
        nfov = np.reshape(np.round(AA + BB + CC + DD).astype(np.uint8), [self.height, self.width, 3])
        import matplotlib.pyplot as plt
        plt.imshow(nfov)
        plt.show()
        return nfov

    def toNFOV(self, frame, center_point): # Frame = equibild
        self.frame = frame
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.frame_channel = frame.shape[2]

        self.cp = self._get_coord_rad(center_point=center_point, isCenterPt=True)

        # Hier scheint eine Art der Konvertierung vorzuliegen -> Alles zwischen PI und Pi / 2 aber noch keine Verzerrungsberechnung
        convertedScreenCoord = self._get_coord_rad(isCenterPt=False)

        # Hier werden die Koordinaten erst wirklich aufgespreizt. sprich nach meinem Verständnis sind diese sperical Coords die Coords die wir brauchen
        spericalCoord = self._calcSphericaltoGnomonic(convertedScreenCoord)

        # Und DANACH wird erst interpoliert!
        return self._bilinear_interpolation(spericalCoord)

    def backToEqui(self, img, bbox, center_point):
        self.cp = self._get_coord_rad(center_point=center_point, isCenterPt=True)
        # Scale factor is img width / output_img_width ...
        # scale_x = img.shape[1] / self.width
        # scale_y = img.shape[0] / self.height
        # Scaling between 0 - 1
        # bbox =((bbox[0] / np.array([self.width, self.height])), (bbox[1] / np.array([self.width, self.height])))

        # bbox = list(bbox) # change to [(), ()]
        # bbox_scaled = (tuple(bbox[0] * np.array([scale_x, scale_y])), tuple(bbox[1] * np.array([scale_x, scale_y])))
        screen_points = self._get_coord_rad_customData(bbox)

        sphericalCoord = self._calcSphericaltoGnomonic(screen_points)
        # interpolate img to get bbox expressions between 0 and 1

        x_axis_inter = interp1d([0, 1], [0, img.shape[1]])
        y_axis_inter = interp1d([0, 1], [0, img.shape[0]])
        # print(sphericalCoord[1][1])
        # print('punkt1')
        # print(x_axis_inter((sphericalCoord[0][0])))
        # print(y_axis_inter((sphericalCoord[0][1])))
        # print('punkt2')
        # print(x_axis_inter((sphericalCoord[1][0])))
        # print(y_axis_inter((sphericalCoord[1][1])))
        # print('meep')

        return (x_axis_inter((sphericalCoord[0][0])), y_axis_inter((sphericalCoord[0][1]))), (x_axis_inter((sphericalCoord[1][0])), y_axis_inter((sphericalCoord[1][1])))


# test the class
if __name__ == '__main__':
    out_height = 400
    out_width = 800
    import imageio as im
    img = im.imread('images/360.jpg')
    print(f'shape of input image: {img.shape}') # zuerst höhe dann breite
    nfov = NFOV(height=out_height, width=out_width)
    center_point = np.array([0.5, 0.5])  # camera center point (valid range [0,1])
    projected_img = nfov.toNFOV(img, center_point)

    # Scale factor is img width / output_img_width ...
    # scale_x = img.shape[1] / out_width
    # scale_y = img.shape[0] / out_height

    # BB definieren als tuple

    # bbox = [[
    #     (100, 200)],
    #     [(105, 205)
    # ]]


    bbox0 = [400 / out_width, 150 / out_height] # bild sample nvov -> fenster links oben / rechts unten
    bbox1 = [575 / out_width, 225 / out_height]
    bbox = np.array(([bbox0, bbox1])).T
    # bbox = np.array([])
    # bbox = np.append(bbox, np.array([100, 200]).T)
    # bbox = np.append(bbox, np.array([105, 205]).T) # (x1,y1), (x2,y2)

    # (customCoords * 2 - 1) * np.array([self.PI, self.PI_2]) * (
    #         np.ones(customCoords.shape) * self.FOV)
    # print('bbox scaled ' + bbox * 1.5)
    nfovback = NFOV(height=out_height, width=out_width)
    backprojectedBBox = nfovback.backToEqui(img,  bbox=bbox, center_point= center_point)
    # print(f'shape of backprojectedBox: {backprojectedBBox.shape}')
    # nfov._get_coord_rad_customData()
    # calcSphericalToGnonomic
    # ergebnis muss 4 zahlen im Bereich 0 -1 und im equirectangularraum
    # und mal bildgröße und bildweite -> kreis
    # for schleife mit ganzem rechteck -> müsste verzerrt sein

    # Plotting the bbox coords on the projected img
    # circle_bbox0 = plt.Circle((380, 140), 15, color='yellow')  # Generating the circle
    # circle_bbox1 = plt.Circle((525, 235), 15, color='red')  # Generating the circle
    # fig, ax = plt.subplots()  # note we must use plt.subplots, not plt.subplot
    # ax.add_artist(circle_bbox0)  # add the circle
    # ax.add_artist(circle_bbox1)  # add the circle
    # plt.imshow(projected_img, origin='upper')  # Plotting the point
    # plt.title('BBOX Coords in planar')
    # plt.show()

    # plotting a rectangle

    bboxTopLeft = (380, 140)
    bboxBottomRight = (525, 235)


    lengthX = bboxBottomRight[0] - bboxTopLeft[0]
    lengthY = bboxBottomRight[1] - bboxTopLeft[1]

    linx = np.linspace(380, 525, 525-380, True) # eher vll mti np.arrange
    print(linx)
    # print(lengthX)
    # print(lengthY)

    # Hier alle für die Linie oben und unten
    topLine = []
    # for x in range(lengthX + 1):
    #     if x != 0:
    #         print(bboxTopLeft[0] + x, bboxTopLeft[1])
    #         bboxTopLine = [(bboxTopLeft[0] + x) / out_width, bboxTopLeft[1] / out_height]
    #         bboxTopLine = np.array(([bboxTopLine])).T
    #         topLine.append(bboxTopLine)
            # print(bboxTopLeft[0] + x, bboxBottomRight[1])
    # Hier alle für die Linie links und rechts
    # for y in range(lengthY + 1):
    #     if y != 0:
    #         print(bboxTopLeft[0], bboxTopLeft[1] + y)
    #         # print(bboxBottomRight[0], bboxBottomRight[1] + y)


    # print(nfovback.backToEqui(img,  bbox=topLine, center_point=center_point))
    # line = []
    # steps = 30
    # fig1, ax = plt.subplots()  # note we must use plt.subplots, not plt.subplot
    # for step in range(1, steps+1):
    #     bbox0 = [100 / out_width, (100 + step*2) / out_height]  # bild sample nvov -> fenster links oben / rechts unten
    #     bbox1 = [575 / out_width, (100 + step*2) / out_height]
    #     # bbox0 = [(100 + 50 * step) / out_width, 100 / out_height]  # bild sample nvov -> fenster links oben / rechts unten
    #     # bbox1 = [(575 + 25 * step) / out_width, 100 / out_height]
    #     # print(bbox0)
    #     # print(bbox1)
    #     bbox = np.array(([bbox0, bbox1])).T
    #     # print(bbox)
    #     line.append(nfovback.backToEqui(img,  bbox=bbox, center_point=center_point))
    #     # print(line)
    # for backprojectedBBox in line:
    #     # print(backprojectedBBox)
    #     circle_bbox0_equi = plt.Circle((backprojectedBBox[0]), 15, color='magenta')  # Generating the circle
    #     circle_bbox1_equi = plt.Circle((backprojectedBBox[1]), 15, color='yellow')  # Generating the circle
    #     ax.add_artist(circle_bbox0_equi)  # add the circle
    #     ax.add_artist(circle_bbox1_equi)  # add the circle
    #
    # plt.imshow(img, origin='upper')  # Plotting the point
    # plt.title('BBOX Coords in Equirectangular')
    # plt.show()


    print('Efefe')