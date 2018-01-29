import math

def cat_to_num(cat_list, ser):
    print("In")
    for i in range(0, len(ser)):
        # print(i)
        for j in range(0, len(cat_list)):
            # print(ser[i])
            if ser[i] == cat_list[j]:
                # ser.iat[i] = j + 1
                ser.iat[i] = len(cat_list) - j
                break
            elif cat_list[j] == 'NA':
                if math.isnan(ser.iat[i]):
                    ser.iat[i] = len(cat_list) - j
                    break
    return
