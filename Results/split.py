import pickle

# part_pkl = pickle.load(open("700000_partpair_0.8_0.3.pkl", "rb"), encoding='bytes')  # the best partpair score
# part1 = {}
# part2 = {}
# part3 = {}
# part4 = {}
# for key, value in part_pkl.items():
#     print(key)
#     if key <=2500:
#         part1[key] = part_pkl[key]
#     elif key<=5000:
#         part2[key] = part_pkl[key]
#     elif key<=7500:
#         part3[key] = part_pkl[key]
#     else:
#         part4[key] = part_pkl[key]
#
# output1 = open('700000_partpair_0.8_0.3_part1.pkl', 'wb')
# pickle.dump(part1, output1)
# output2 = open('700000_partpair_0.8_0.3_part2.pkl', 'wb')
# pickle.dump(part2, output2)
# output3 = open('700000_partpair_0.8_0.3_part3.pkl', 'wb')
# pickle.dump(part3, output3)
# output4 = open('700000_partpair_0.8_0.3_part4.pkl', 'wb')
# pickle.dump(part4, output4)

tin_pkl = pickle.load(open("1700000_TIN_HICO_0.8_0.3new_pair_selection_H2O2.pkl", "rb"), encoding='bytes')  # the best partpair score
part1 = {}
part2 = {}
part3 = {}
part4 = {}
for key, value in tin_pkl.items():
    print(key)
    if key <=2500:
        part1[key] = tin_pkl[key]
    elif key<=5000:
        part2[key] = tin_pkl[key]
    elif key<=7500:
        part3[key] = tin_pkl[key]
    else:
        part4[key] = tin_pkl[key]

output1 = open('1700000_TIN_HICO_0.8_0.3new_pair_selection_H2O2_part1.pkl', 'wb')
pickle.dump(part1, output1)
output2 = open('1700000_TIN_HICO_0.8_0.3new_pair_selection_H2O2_part2.pkl', 'wb')
pickle.dump(part2, output2)
output3 = open('1700000_TIN_HICO_0.8_0.3new_pair_selection_H2O2_part3.pkl', 'wb')
pickle.dump(part3, output3)
output4 = open('1700000_TIN_HICO_0.8_0.3new_pair_selection_H2O2_part4.pkl', 'wb')
pickle.dump(part4, output4)