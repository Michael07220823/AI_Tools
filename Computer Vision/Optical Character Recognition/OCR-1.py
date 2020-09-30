#!/usr/bin/env python
# coding: utf-8

# # Optical Charater Recognition

# ## 範例一－中文歌譜

# In[31]:


from PIL import Image
import pytesseract as pytess
import matplotlib.pyplot as plt

img = Image.open("F:/Github/AI_Tools/Computer Vision/Optical Character Recognition/images/song_sheet/隱形的翅膀.png")
plt.figure("Song sheet")
plt.imshow(img)
plt.show()

print(pytess.image_to_string(img, lang='chi_tra'))


# ## 範例二－英文文章

# In[33]:


import cv2
import pytesseract as pytess
import matplotlib.pyplot as plt

img = cv2.imread("F:/Github/AI_Tools/Computer Vision/Optical Character Recognition/images/article/article-2.jpg")
plt.figure("english article")
plt.imshow(img)
plt.show()

print(pytess.image_to_string(img, lang='eng'))


# ## 範例三－數學方程式

# In[38]:


import cv2
import pytesseract as pytess
import matplotlib.pyplot as plt

img = cv2.imread("F:/Github/AI_Tools/Computer Vision/Optical Character Recognition/images/math/formula-1.jpg")
plt.figure("math")
plt.imshow(img)
plt.show()

print(pytess.image_to_string(img, lang='eng'))

