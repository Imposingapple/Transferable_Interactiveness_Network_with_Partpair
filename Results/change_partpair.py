import pickle
my_pkl = pickle.load( open('700000_partpair_0.8_0.3.pkl', "rb" ) ,encoding='bytes')
sum=0
for key,value in my_pkl.items():
    for ele in value:
        # print(ele)
        sum+=1
        ele[3]=ele[7]
        ele=ele[:7]
        # print(ele)
        # print()

print(sum)

output = open('700000_partpair_0.8_0.3_revised.pkl', 'wb')
pickle.dump(my_pkl, output)