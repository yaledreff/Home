color_map = {
 '0': [0, 0, 0],
 '1': [153, 153, 0],
 '2': [255, 204, 204],
 '3': [255, 0, 127],
 '4': [0, 255, 0],
 '5': [0, 204, 204],
 '6': [255, 0, 0],
 '7': [0, 0, 255]
}

def getColoredMask(image):
 sourcesColors = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5), (6, 6, 6), (7, 7, 7)]
 targetColors = [(0, 0, 0), (153, 153, 0), (255, 204, 204), (255, 0, 127), (0, 255, 0), (0, 204, 204), (255, 0, 0), (0, 0, 255)]

 imageSAV = image

 for i in range(0, 8):
 # i=7
  print(i)
  sr, sg, sb = sourcesColors[i]
  print(sourcesColors[i])
  tr, tg, tb = targetColors[i]
  print(targetColors[i])
  red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
  mask = (red == sr) & (green == sg) & (blue == sb)
  image[:, :, :3][mask] = [tb, tg, tr]
 return imageSAV
