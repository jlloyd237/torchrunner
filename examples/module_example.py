from torchrunner.module import conv_net, res_net, dense_net

cnn = conv_net(ni=3, no=10, nf=[16, 32, 64], nh=[32, 32])
print(cnn)

rn = res_net(ni=3, no=10, nf=[16, 32, 64], nh=[32, 32])
print("----------")
print(rn)

dn = dense_net(ni=3, no=10, nf=[16, 32, 64], nh=[32, 32])
print("----------")
print(dn)