import math

def cat_to_num(cat_list, ser):
    ser2 = ser
    print("In")
    for i in range(0, len(ser2)):
        # print(i)
        for j in range(0, len(cat_list)):
            # print(ser2[i])
            if ser2[i] == cat_list[j]:
                # ser2.iat[i] = j + 1
                ser2.iat[i] = len(cat_list) - j
                break
            elif cat_list[j] == 'NA':
                if math.isnan(ser2.iat[i]):
                    ser2.iat[i] = len(cat_list) - j
                    break
    return
