import cv2
import math
import matplotlib.pyplot as plt
import time
start = time.time()

img = cv2.imread(r'C:\Users\diana\OneDrive\Desktop\Facultate\Python\Licenta\noise.jpg')
img2 = cv2.imread(r'C:\Users\diana\OneDrive\Desktop\Facultate\Python\Licenta\noise.jpg')

# img = cv2.imread(r'C:\Users\diana\OneDrive\Desktop\Facultate\Python\Licenta\apple.png')
# img2 = cv2.imread(r'C:\Users\diana\OneDrive\Desktop\Facultate\Python\Licenta\apple.png')

# img = cv2.imread(r'C:\Users\diana\OneDrive\Desktop\Facultate\Python\Licenta\Noise_salt_and_pepper.png')
# img2 = cv2.imread(r'C:\Users\diana\OneDrive\Desktop\Facultate\Python\Licenta\Noise_salt_and_pepper.png')

# img = cv2.imread(r'C:\Users\diana\OneDrive\Desktop\Facultate\Python\Licenta\image_denoising-1.png')
# img2 = cv2.imread(r'C:\Users\diana\OneDrive\Desktop\Facultate\Python\Licenta\image_denoising-1.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

h = img.shape[0]
w = img.shape[1]

# img = cv2.resize(img, (int(w/2), int(h/2)))
# img2 = cv2.resize(img2, (int(w/2), int(h/2)))
#
# h = img.shape[0]
# w = img.shape[1]

print(h, w)

d = 35
n = 3
m = 2
m_prim = 1
non_corrupted = list()
non_diagnosed = list()
corrupted = list()
dictMF = {}
dictXi = {}
dictNrNonCor = {}

def peerGroup(xi, d, n):
    dictNrNonCor[str(xi[0]) + str(xi[1])] = 0
    P = list() # peer group
    R = list() # Red
    G = list() # Green
    B = list() # Blue
    for i in range(xi[0] - 1, xi[0] + n - 1):
        for j in range(xi[1] - 1, xi[1] + n - 1):
            R.append(img[i, j][0])
            G.append(img[i, j][1])
            B.append(img[i, j][2])
            dictMF[str(i)+str(j)] = str(xi[0]) + str(xi[1])
            if ((xi[0], xi[1])!=(i, j)):
                euclidian_dist = math.sqrt(sum([(int(a) - int(b)) ** 2 for a, b in zip(img[xi[0], xi[1]], img[i, j])]))
                if euclidian_dist <= d:
                    P.append((i, j))
    R.sort()
    G.sort()
    B.sort()
    dictXi[str(xi[0]) + str(xi[1])] = (R[int(n*n/2)], G[int(n*n/2)], B[int(n*n/2)])
    return P

def diagnose(P, xi, n, corrupted, non_corrupted, non_diagnosed):

    if len(P) >= (m + 1):
        for i in range(xi[0] - 1, xi[0] + n - 1):
            for j in range(xi[1] - 1, xi[1] + n - 1):
                if ((xi[0], xi[1]) != (i, j)):
                    if (i, j) in P:
                        non_corrupted.append((i, j))
                        dictNrNonCor[str(xi[0]) + str(xi[1])] += 1
                    else:
                        non_diagnosed.append((i, j))
    else:
        corrupted.append(xi)
        img[xi[0], xi[1]][0] = dictXi[dictMF[str(xi[0]) + str(xi[1])]][0]
        img[xi[0], xi[1]][1] = dictXi[dictMF[str(xi[0]) + str(xi[1])]][1]
        img[xi[0], xi[1]][2] = dictXi[dictMF[str(xi[0]) + str(xi[1])]][2]
        for i in range(xi[0] - 1, xi[0] + n - 1):
            for j in range(xi[1] - 1, xi[1] + n - 1):
                if ((xi[0], xi[1]) != (i, j)):
                    if (i, j) not in P:
                        non_diagnosed.append((i, j))

def rediangose(corrupted, non_corrupted, non_diagnosed):
    for x in non_diagnosed:
        xi = x
        P = peerGroup(xi, d, n)
        if dictNrNonCor[str(xi[0]) + str(xi[1])] == m_prim:
            non_corrupted.append(xi)
        elif len(P) >= (m + 1):
            for i in range(xi[0] - 1, xi[0] + n - 1):
                for j in range(xi[1] - 1, xi[1] + n - 1):
                    if ((xi[0], xi[1]) != (i, j)):
                        if (i, j) in P:
                            non_corrupted.append((i, j))
        else:
            corrupted.append(xi)
            img[xi[0], xi[1]][0] = dictXi[dictMF[str(xi[0]) + str(xi[1])]][0]
            img[xi[0], xi[1]][1] = dictXi[dictMF[str(xi[0]) + str(xi[1])]][1]
            img[xi[0], xi[1]][2] = dictXi[dictMF[str(xi[0]) + str(xi[1])]][2]


for i in range(0, int(h) - n, n):
    for j in range(0, int(w) - n, n):
        xi = (i, j)
        P = peerGroup(xi, d, n)
        diagnose(P, xi, n, corrupted, non_corrupted, non_diagnosed)

# for xi in corrupted:
#     nn = n
#     while (dictNrNonCor[str(xi[0]) + str(xi[1])] == 0) and nn < h:
#         print(xi)
#         nn += 2
#         if (xi[0] + nn < h) and (xi[1] + nn < w):
#             P = peerGroup(xi, d, nn)
#         print(nn)
#         diagnose(P, xi, nn, corrupted, non_corrupted, non_diagnosed)

corrupted = list(dict.fromkeys(corrupted))
corrupted = [x for x in corrupted if x[0] != -1 and x[1] != -1]
corrupted.sort()
non_corrupted = [x for x in non_corrupted if x[0] > 0 and x[1] > 0]
non_corrupted.sort()
non_diagnosed = [x for x in non_diagnosed if x[0] > 0 and x[1] > 0]
non_diagnosed.sort()

rediangose(corrupted, non_corrupted, non_diagnosed)

# P = peerGroup((1, 1), d, n)
# diagnose(P, (1, 1), n, corrupted, non_corrupted, non_diagnosed)
# rediangose(corrupted, non_corrupted, non_diagnosed)
# for i in range(0, 15):
#     print("Corrupted:", corrupted[i], non_corrupted[i])


corrupted = list(dict.fromkeys(corrupted))
corrupted = [x for x in corrupted if x[0]!= -1 and x[1]!=-1]
corrupted.sort()
non_corrupted = [x for x in non_corrupted if x[0]>0 and x[1]>0]
non_corrupted.sort()
non_diagnosed = [x for x in non_diagnosed if x[0]>0 and x[1]>0]
non_diagnosed.sort()
# amf(corrupted, non_corrupted, dictAMF)

# for xi in corrupted:
#     sumRGB = [0, 0, 0, 0]  # R, G, B, number of non-corrupted in that P
#     for i in range(xi[0] - 1, xi[0] + n - 1):
#         for j in range(xi[1] - 1, xi[1] + n - 1):
#             x = (i, j)
#             if x in non_corrupted:
#                 print(x, "in non_corrupted")
#                 sumRGB[0] += img[x[0], x[1]][0]
#                 sumRGB[1] += img[x[0], x[1]][1]
#                 sumRGB[2] += img[x[0], x[1]][2]
#                 sumRGB[3] += 1
#     print("suma pentru: ", xi, "esteee: ", sumRGB)
#     dictAMF[str(xi[0]) + str(xi[1])] = sumRGB




# print("Cccc: ", corrupted)
# print("Nccc: ", non_corrupted)
# print("Nddd: ", non_diagnosed)
# print("MF: ", dictMF)
#
# print("xi:", dictXi)
# print(peerGroup((0, 2), d, n))
# print("ccccccccccc")
# print("Non-corrupted:", non_corrupted)
# print(len(non_corrupted))
# print("Non-diagnosed:", non_diagnosed)
# print("AMF:", dictAMF)

# for x in range(0, 4):
#     for y in range(0, 4):
#         print("(" + str(x) + "," + str(y) + ")", img[x, y])


# for x in corrupted:
#     img[x[0], x[1]][0] = dictXi[dictMF[str(x[0]) + str(x[1])]][0]
#     img[x[0], x[1]][1] = dictXi[dictMF[str(x[0]) + str(x[1])]][1]
#     img[x[0], x[1]][2] = dictXi[dictMF[str(x[0]) + str(x[1])]][2]

# for x in corrupted:
#     img[x[0], x[1]][0] = 255
#     img[x[0], x[1]][1] = 0
#     img[x[0], x[1]][2] = 204

print(len(corrupted))
end = time.time()
print(end - start)

print(img[6, 246])
print(img[189, 192])
print(img[195, 216])

plt.subplot(121), plt.imshow(img2)
plt.subplot(122), plt.imshow(img)
plt.show()
# #
# print(img.shape[0], img.shape[1])
# print(img[407, 319])

# nemo = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(nemo)
# plt.show()

# print(img[4, 4])
# print(img[4, 4][0])

# print(len(P))

# for i in range(1, n+1):
#     for j in range(1, n+1):
#         print(img[i, j], i, j)

end = time.time()
print(end - start)