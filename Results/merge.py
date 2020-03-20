import pickle
import sys
import os

# merge tin parts
tin_pkl1 = pickle.load(open('1700000_TIN_HICO_0.8_0.3new_pair_selection_H2O2_part1.pkl', "rb"), encoding='bytes')
tin_pkl2 = pickle.load(open('1700000_TIN_HICO_0.8_0.3new_pair_selection_H2O2_part2.pkl', "rb"), encoding='bytes')
tin_pkl3 = pickle.load(open('1700000_TIN_HICO_0.8_0.3new_pair_selection_H2O2_part3.pkl', "rb"), encoding='bytes')
tin_pkl4 = pickle.load(open('1700000_TIN_HICO_0.8_0.3new_pair_selection_H2O2_part4.pkl', "rb"), encoding='bytes')
tin_pkl = {}
for key, value in tin_pkl1.items():
    tin_pkl[key] = tin_pkl1[key]
for key, value in tin_pkl2.items():
    tin_pkl[key] = tin_pkl2[key]
for key, value in tin_pkl3.items():
    tin_pkl[key] = tin_pkl3[key]
for key, value in tin_pkl4.items():
    tin_pkl[key] = tin_pkl4[key]

output = open("1700000_TIN_HICO_0.8_0.3new_pair_selection_H2O2.pkl", 'wb')
pickle.dump(tin_pkl, output)
print("tin parts merged")
for i in range(4):
    os.remove("1700000_TIN_HICO_0.8_0.3new_pair_selection_H2O2_part"+str(i+1)+".pkl")

# merge partpair parts
part_pkl1 = pickle.load(open("700000_partpair_0.8_0.3_part1.pkl", "rb"), encoding='bytes')
part_pkl2 = pickle.load(open("700000_partpair_0.8_0.3_part2.pkl", "rb"), encoding='bytes')
part_pkl3 = pickle.load(open("700000_partpair_0.8_0.3_part3.pkl", "rb"), encoding='bytes')
part_pkl4 = pickle.load(open("700000_partpair_0.8_0.3_part4.pkl", "rb"), encoding='bytes')
part_pkl = {}
for key, value in part_pkl1.items():
    part_pkl[key] = part_pkl1[key]
for key, value in part_pkl2.items():
    part_pkl[key] = part_pkl2[key]
for key, value in part_pkl3.items():
    part_pkl[key] = part_pkl3[key]
for key, value in part_pkl4.items():
    part_pkl[key] = part_pkl4[key]

output = open("700000_partpair_0.8_0.3.pkl", 'wb')
pickle.dump(part_pkl, output)
print("part parts merged")
for i in range(4):
    os.remove("700000_partpair_0.8_0.3_part"+str(i+1)+".pkl")

# sum=0
# tin_pkl = pickle.load(open('1700000_TIN_HICO_0.8_0.3new_pair_selection_H2O2.pkl', "rb"), encoding='bytes')
# for key,value in tin_pkl.items():
#     for ele in value:
#         sum+=1
# print(sum)   # correct value is 466627
