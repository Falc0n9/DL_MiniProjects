
conv_layer = 4*[(1,1)]

def all_conv(lista,index):
    for ks in range(1,5):
        for channels in range(1,5):
            lista[index] = (ks,channels)
            if index == 0:
                yield lista
            else:
                all_conv(lista,index-1)            
            

            
x = all_conv(conv_layer,3)
for item in x:
    print(item)
